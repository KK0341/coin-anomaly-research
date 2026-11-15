"""Cloud B5 training; test evaluation is deliberately a separate final action."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from PIL import Image
import b0_reference as base
from model import Dual
from prepare import digest


def loader(ds, batch, workers, shuffle=False):
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, generator=torch.Generator().manual_seed(42+ds.epoch))


def verify(data):
    record = json.loads((data/'assets.json').read_text('utf8'))
    for name, expected in record['sha256'].items():
        assert digest(data/name) == expected, name
    manifest = json.loads((data/'manifest.json').read_text('utf8'))
    ids = set()
    for split, size in [('train', 1583), ('val', 198), ('test', 198)]:
        assert len(manifest[split]) == size
        for row in manifest[split]:
            coin = row['钱币编号']
            assert coin not in ids
            ids.add(coin)
            assert row['异常代码'] in base.CODES and row['二分类标签'] == '1'
            assert row['images'] == [f'images/{coin}_0.png', f'images/{coin}_1.png']
            for name in row['images']:
                with Image.open(data/name) as im:
                    assert im.size == (512, 512), (name, im.size)
                    im.verify()
    return manifest


@torch.inference_mode()
def predict(model, ds, weights, args, dtype, mode='full'):
    model.eval()
    yy, pp, total, denom = [], [], 0., 0.
    for x, y, _ in loader(ds, args.batch, args.workers):
        with torch.autocast('cuda', dtype=dtype):
            if mode == 'global':
                z = model(x.cuda(), details=True)[1]
            else:
                z = model(x.cuda(), uniform=(mode == 'uniform'))
        z = z.float()
        assert torch.isfinite(z).all()
        y = y.cuda()
        total += nn.functional.cross_entropy(z, y, weight=weights, reduction='sum').item()
        denom += weights[y].sum().item()
        yy.extend(y.cpu().tolist()); pp.extend(z.softmax(1).cpu().tolist())
    return np.array(yy), np.array(pp), total/denom


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch', type=int, choices=[1, 2, 4], default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--amp', choices=['auto', 'bf16', 'fp16'], default='auto')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expose exactly one CUDA GPU'
    assert '5090' not in torch.cuda.get_device_name(), 'Unauthorized 5090 hardware'
    args.out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (args.out/'training.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    base.DATA = args.data
    manifest = verify(args.data)
    base.seed(42)
    m = Dual(); m.initialize_a8(args.data/'a8.pt'); m.cuda()
    bf16 = torch.cuda.is_bf16_supported()
    assert args.amp != 'bf16' or bf16
    dtype = torch.bfloat16 if args.amp == 'bf16' or (args.amp == 'auto' and bf16) else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    opt = torch.optim.AdamW([{'params': m.net.parameters(), 'lr': 1e-5},
                            {'params': [v for k,v in m.named_parameters() if not k.startswith('net.')], 'lr': 1e-4}], weight_decay=.01)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=.5, patience=4,
               threshold=0, threshold_mode='abs', min_lr=[1e-7, 1e-6])
    counts = np.bincount([base.CODES.index(r['异常代码']) for r in manifest['train']], minlength=6)
    weights = 1/np.sqrt(counts); weights /= weights.mean()
    w = torch.tensor(weights, dtype=torch.float32, device='cuda')
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=Path(__file__).parent, text=True).strip()
    cfg = dict(experiment='B5', git_commit=commit, gpu=torch.cuda.get_device_name(), amp=str(dtype),
               batch=args.batch, accumulation=8//args.batch, effective_batch=8, resolution=512,
               seed=42, max_epochs=60, patience=10, class_counts=counts.tolist(), class_weights=weights.tolist(),
               backbone_lr=1e-5, head_lr=1e-4, weight_decay=.01, clip_norm=5,
               parameters=sum(v.numel() for v in m.parameters()), added_parameters=9222,
               initialization='Original A8 backbone and hidden MLP; seeded new six-class output',
               selection='validation Macro-F1; earliest epoch wins ties',
               asset_manifest_sha256=digest(args.data/'assets.json'),
               source_sha256={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')})
    config_path = args.out/'config.json'
    if args.resume:
        assert json.loads(config_path.read_text('utf8')) == cfg, 'Resume config/source/hardware mismatch'
    elif config_path.exists():
        raise FileExistsError('Use a new output directory or --resume')
    else:
        base.save(config_path, cfg)
    (args.out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'], text=True), 'utf8')
    (args.out/'nvidia-smi.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True), 'utf8')
    ds = base.Coins(manifest['train'], True); val = base.Coins(manifest['val'])
    history, best, stale, start = [], -1., 0, 1
    if args.resume:
        ck = torch.load(args.out/'last.pt', map_location='cpu', weights_only=False)
        m.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler'])
        history, best, stale, start = ck['history'], ck['best'], ck['stale'], ck['epoch']+1
    for epoch in range(start, 61):
        if stale >= 10: break
        base.seed(42+epoch); ds.epoch = epoch
        dl = loader(ds, args.batch, args.workers, True); it = iter(dl)
        m.train(); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); begun = time.perf_counter()
        total, denom, yy, pp, updates = 0., 0., [], [], 0
        accum = 8//args.batch
        for offset in range(0, len(dl), accum):
            batches = [next(it) for _ in range(min(accum, len(dl)-offset))]
            group_den = sum(w[y.cuda()].sum().item() for _,y,_ in batches)
            opt.zero_grad(set_to_none=True)
            for x,y,_ in batches:
                y = y.cuda(non_blocking=True)
                with torch.autocast('cuda', dtype=dtype): z = m(x.cuda(non_blocking=True))
                loss = nn.functional.cross_entropy(z.float(), y, weight=w, reduction='sum')
                assert torch.isfinite(loss)
                scaler.scale(loss/group_den).backward()
                total += loss.item(); denom += w[y].sum().item()
                yy.extend(y.cpu().tolist()); pp.extend(z.detach().float().softmax(1).cpu().tolist())
            scaler.unscale_(opt)
            grad = nn.utils.clip_grad_norm_(m.parameters(), 5, error_if_nonfinite=True)
            assert grad > 0
            if args.smoke:
                assert m.queries.grad is not None and m.queries.grad.abs().sum() > 0
                before = m.queries.detach().clone()
            scaler.step(opt); scaler.update(); updates += 1
            if args.smoke:
                assert not torch.equal(before, m.queries.detach()), 'No optimizer update'
            if args.smoke and updates == 8:
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-begun
                base.save(args.out/'smoke.json', dict(cfg, loss=total/denom, updates=updates,
                    grad_norm=float(grad), seconds=elapsed, coins_per_second=len(yy)/elapsed,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))
                print('SMOKE PASS', flush=True); return
        vy,vp,vl = predict(m, val, w, args, dtype)
        met = base.metrics(vy,vp); f = met['macro_f1']
        if f > best:
            best, stale = f, 0
            base.atomic({'model':m.state_dict(), 'epoch':epoch, 'val_macro_f1':f, 'config':cfg}, args.out/'best.pt')
        else: stale += 1
        sched.step(f); torch.cuda.synchronize()
        h = dict(epoch=epoch, train_loss=total/denom, val_loss=vl,
                 train_macro_f1=base.metrics(np.array(yy),np.array(pp))['macro_f1'], val_macro_f1=f,
                 val_accuracy=met['accuracy'], val_balanced_accuracy=met['balanced_accuracy'],
                 seconds=time.perf_counter()-begun, stale=stale, backbone_lr=opt.param_groups[0]['lr'],
                 peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                 peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)
        history.append(h)
        base.atomic(dict(model=m.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                    scaler=scaler.state_dict(), epoch=epoch, history=history, best=best, stale=stale,
                    config=cfg, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), args.out/'last.pt')
        base.save(args.out/'history.json', history); base.csvwrite(args.out/'history.csv', history)
        print(json.dumps(h), flush=True)
    base.save(args.out/'TRAINING_COMPLETE.json', {'best_val_macro_f1': best, 'epochs':len(history)})

if __name__ == '__main__':
    main()
