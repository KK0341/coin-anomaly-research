"""Linux A1 path adapter. Default is asset validation only. Original trainer stays unchanged."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PureWindowsPath
import sys

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / '算法训练/A/A1/代码/train.py'
PREFIX = PureWindowsPath('E:/钱币鉴别')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()

def mapped_path(value, root):
    relative = PureWindowsPath(value).relative_to(PREFIX)
    target = root.joinpath(*relative.parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError('Image path escapes image root')
    return target

def validate(args, cfg):
    mapping = {}
    counts = {}
    for split, name in [('train', '训练集.csv'), ('val', '验证集.csv'), ('test', '测试集.csv')]:
        path = args.splits / name
        if digest(path) != cfg['split_sha256'][split]:
            raise ValueError(f'Split checksum mismatch: {split}; keep original CSV bytes')
        with path.open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        counts[split] = len(rows)
        for row in rows:
            for side in ['正面路径', '背面路径']:
                target = mapped_path(row[side], args.images)
                if not target.is_file():
                    raise FileNotFoundError(target)
                mapping[row[side]] = str(target)
    if not digest(args.assets / 'A0/模型/convnext_tiny-983f1562.pth').startswith('983f1562'):
        raise ValueError('Pretrained weight checksum mismatch')
    print(json.dumps({'asset_check': 'passed', 'split_rows': counts, 'mapped_images': len(mapping)}, ensure_ascii=False))
    return mapping

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True, help='Linux directory corresponding to E:/钱币鉴别')
    parser.add_argument('--assets', type=Path, required=True, help='Contains A0/模型/official-weight.pth')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0', help='One visible CUDA device ID or UUID')
    parser.add_argument('--execute', action='store_true', help='Actually start training, including original final evaluation')
    parser.add_argument('--resume', action='store_true', help='Continue this adapter run from its last.pt')
    args = parser.parse_args()
    if ',' in args.gpu or not args.gpu.strip():
        parser.error('Select exactly one GPU')
    for field in ['splits', 'images', 'assets', 'output']:
        setattr(args, field, getattr(args, field).resolve())
    if any(args.output.is_relative_to(parent) for parent in [REPO, args.images, args.splits, args.assets]):
        parser.error('Use a separate output directory outside the checkout and data directories')
    cfg = json.loads((REPO / 'cloudstudio/a1_config.json').read_text(encoding='utf-8'))
    mapping = validate(args, cfg)
    if not args.execute:
        print('Check only: no import of trainer, no GPU computation, no output writes.')
        return
    if sys.platform != 'linux':
        parser.error('Execution adapter requires Linux; local validation remains available')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import fcntl
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one usable CUDA GPU required; no silent CPU fallback')
    # One lock per user and selected device, shared across output directories.
    lock_path = Path.home() / ('.coin-cloud-gpu-' + hashlib.sha256(args.gpu.encode()).hexdigest()[:16] + '.lock')
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {'source_sha256': digest(SOURCE), 'config_sha256': digest(REPO / 'cloudstudio/a1_config.json')}
        receipt = args.output / 'cloud_run.json'
        if args.resume:
            if json.loads(receipt.read_text(encoding='utf-8')) != identity:
                raise RuntimeError('Source/config changed since this run')
            if not (args.output / '模型/last.pt').is_file() or not (args.output / '模型/best.pt').is_file():
                raise RuntimeError('Resume needs both last.pt and best.pt')
        else:
            if args.output.exists() and any(args.output.iterdir()):
                raise RuntimeError('New run needs an empty output; use --resume for existing runs')
        for name in ['模型', '结果']:
            (args.output / name).mkdir(parents=True, exist_ok=True)
        if not args.resume:
            (args.output / '模型/config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
            receipt.write_text(json.dumps(identity, indent=2), encoding='utf-8')
        # Import unchanged source; cloud checkout must not include historical loader cache selections.
        if (SOURCE.parents[1] / '结果/loader_selection.json').exists():
            raise RuntimeError('Use a clean code checkout without historical loader_selection.json')
        spec = importlib.util.spec_from_file_location('coin_a1_original', SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        module.ROOT = args.assets / 'A1'
        module.OUT = args.output / '结果'
        module.MODEL = args.output / '模型'
        module.SPLIT = args.splits
        module.CACHE_MAP = mapping
        module.LOADER_OPTIONS = {k: v for k, v in cfg['loader'].items() if k != 'cache_manifest'}
        # Original train() performs checkpoint restore, per-epoch reseeding and atomic saves.
        print(json.dumps({'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'resume': args.resume}))
        # Defer SIGINT/SIGTERM until the original last.pt atomic save completes.
        import signal
        stopping = False
        class StopAfterCheckpoint(Exception):
            pass
        def request_stop(signum, frame):
            nonlocal stopping
            stopping = True
            print('Stop requested: waiting for next complete epoch checkpoint.', flush=True)
        original_save = module.atomic_save
        def save_with_stop(state, path):
            original_save(state, path)
            if stopping and path.name == 'last.pt':
                raise StopAfterCheckpoint()
        module.atomic_save = save_with_stop
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        print(f'TRAIN_PID={os.getpid()}', flush=True)
        try:
            module.train()
        except StopAfterCheckpoint:
            print('Safely stopped after last.pt save; restart with --resume.', flush=True)

if __name__ == '__main__':
    main()
