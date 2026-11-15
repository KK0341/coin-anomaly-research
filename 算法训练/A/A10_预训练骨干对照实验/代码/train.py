"""A1: shared ConvNeXt-Tiny, concatenated pooled features + MLP, coin-level BCE. Run --mode bench/train."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, csv, gc, hashlib, json, random, time, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms as T
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, precision_recall_curve

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'结果'; MODEL=ROOT/'模型'; SPLIT=Path(r'E:\异常检测\数据划分')
FILES={'train':'训练集.csv','val':'验证集.csv'}
torch.set_num_threads(4)
CACHE_MAP={}
LOADER_OPTIONS={}
if (ROOT/'结果/loader_selection.json').exists():
    LOADER_OPTIONS=json.loads((ROOT/'结果/loader_selection.json').read_text(encoding='utf-8'))
    if LOADER_OPTIONS.get('cache_manifest'):
        CACHE_MAP=json.loads(Path(LOADER_OPTIONS['cache_manifest']).read_text(encoding='utf-8'))

def save_json(p,x):
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def rows(split):
    with open(SPLIT/FILES[split],encoding='utf-8-sig',newline='') as f:
        return [{k:r[k] for k in ['钱币编号','正面路径','背面路径','二分类标签','原始评级','异常代码','正常数字等级']} for r in csv.DictReader(f)]

class Coins(Dataset):
    def __init__(self, split, augment=False):
        self.rows=rows(split)
        self.cache_map=CACHE_MAP.copy()
        self.epoch=torch.zeros(1,dtype=torch.int64).share_memory_()
        self.augment=augment
        # No crop, blur, color jitter or erasing: preserve surface evidence.
        ops=[T.Resize((504,504),antialias=True)]
        if augment: ops += [T.RandomRotation(10,interpolation=T.InterpolationMode.BILINEAR)]
        self.transform=T.Compose(ops+[T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; images=[]
        if self.augment:
            torch.random.default_generator.manual_seed(42+int(self.epoch[0])*100000+i)
        for side in ['正面路径','背面路径']:
            with Image.open(self.cache_map.get(r[side],r[side])) as im: images.append(self.transform(im.convert('RGB')))
        return torch.stack(images),float(r['二分类标签']),i

def worker_init(worker):
    s=torch.initial_seed()%2**32; random.seed(s); np.random.seed(s); torch.set_num_threads(1)

def loader(ds,batch,shuffle=False,workers=None):
    workers=LOADER_OPTIONS.get('workers',2) if workers is None else workers
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=workers,
        pin_memory=LOADER_OPTIONS.get('pin_memory',True),
        persistent_workers=workers>0 and LOADER_OPTIONS.get('persistent_workers',True),worker_init_fn=worker_init,
        **({'prefetch_factor':LOADER_OPTIONS.get('prefetch_factor',2)} if workers else {}))

class Dual(nn.Module):
    def __init__(self):
        super().__init__()
        source=json.loads((ROOT/'来源/provenance.json').read_text(encoding='utf8'))
        self.net=torch.hub.load(source['source_dir'],'dinov2_vits14',source='local',pretrained=False)
        self.head=nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.Linear(256,1))
    def forward(self,x):
        features=self.net(x.flatten(0,1)).reshape(-1,768)
        z=self.head(features).squeeze(1)
        return z,z[:,None]

def pretrained():
    source=json.loads((ROOT/'来源/provenance.json').read_text(encoding='utf8'))
    p=MODEL/'dinov2_vits14_pretrain.pth'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256']
    state=torch.load(p,map_location='cpu',weights_only=True)
    m=Dual(); result=m.net.load_state_dict(state,strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert all(torch.equal(m.net.state_dict()[k],v) for k,v in state.items())
    assert all(p.requires_grad for p in m.net.parameters())
    save_json(OUT/'weight_loading.json',dict(strict=True,keys=len(state),all_tensors_equal=True,
        backbone_parameters=sum(p.numel() for p in m.net.parameters()),total_parameters=sum(p.numel() for p in m.parameters()),
        feature_per_side=384,concat=768,all_backbone_requires_grad=True,sha256=source['sha256']))
    return m.cuda()


def optimizer(m):
    return torch.optim.AdamW([{'params':m.net.parameters(),'lr':2e-5},
                             {'params':m.head.parameters(),'lr':1e-4}],weight_decay=.01)


def memory():
    return {'allocated_MiB':torch.cuda.max_memory_allocated()/2**20,'reserved_MiB':torch.cuda.max_memory_reserved()/2**20}

def train_epoch(m,opt,scaler,dl,accum=1,limit=None):
    m.train(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize()
    start=time.perf_counter(); wait=0.; total=0.; n=0; retries=0
    steps=min(len(dl),limit or len(dl)); it=iter(dl)
    for group_start in range(0,steps,accum):
        group_end=min(group_start+accum,steps); batches=[]
        t=time.perf_counter()
        for j in range(group_start,group_end): batches.append(next(it))
        wait+=time.perf_counter()-t
        group_coins=sum(len(y) for _,y,_ in batches)
        group_rng=torch.cuda.get_rng_state()
        for attempt in range(12):
            opt.zero_grad(set_to_none=True)
            if attempt: torch.cuda.set_rng_state(group_rng)
            group_loss=0.
            for x_cpu,y_cpu,_ in batches:
                x=x_cpu.cuda(non_blocking=True); y=y_cpu.cuda(non_blocking=True).float()
                with torch.autocast('cuda',dtype=torch.float16):
                    z,_=m(x); loss=nn.functional.binary_cross_entropy_with_logits(z,y)
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
                scaler.scale(loss*len(y)/group_coins).backward()
                group_loss+=loss.detach().item()*len(y)
            scaler.unscale_(opt)
            norm=nn.utils.clip_grad_norm_(m.parameters(),5.)
            if not torch.isfinite(norm):
                # Do not update model/Adam state or skip training examples. Reset
                # GradScaler's per-optimizer state and replay this entire group.
                new_scale=max(float(scaler.get_scale())/2.,1.)
                scaler.update(new_scale=new_scale); retries+=1
                print(f'AMP_RETRY step={group_start+1} scale={new_scale} attempt={attempt+1}',flush=True)
                continue
            scaler.step(opt); scaler.update(); break
        else: raise FloatingPointError('Gradient remained nonfinite after 12 lower-scale retries')
        opt.zero_grad(set_to_none=True)
        total+=group_loss; n+=group_coins
        torch.cuda.synchronize()
        if group_end%100==0: print(f'  steps {group_end}/{steps}, coins {n}, seconds {time.perf_counter()-start:.1f}',flush=True)
    elapsed=time.perf_counter()-start
    return {'loss':total/n,'seconds':elapsed,'coins_per_second':n/elapsed,'data_wait_seconds':wait,'data_wait_fraction':wait/elapsed,**memory(),'amp_retries':retries,'amp_final_scale':float(scaler.get_scale())}

@torch.inference_mode()
def predict(m,dl):
    m.eval(); all_y=[]; all_z=[]; all_s=[]
    for x,y,_ in dl:
        with torch.autocast('cuda',dtype=torch.float16): z,s=m(x.cuda(non_blocking=True))
        all_y.extend(y.numpy()); all_z.extend(z.float().cpu().numpy()); all_s.extend(s.float().cpu().numpy())
    z=np.array(all_z); s=np.array(all_s)
    return np.array(all_y,dtype=int),1/(1+np.exp(-z)),1/(1+np.exp(-s)),z,s

def metrics(y,p,threshold=.5):
    pred=p>=threshold
    return {'accuracy':accuracy_score(y,pred),'precision':precision_score(y,pred,zero_division=0),
            'recall':recall_score(y,pred,zero_division=0),'f1':f1_score(y,pred,zero_division=0),
            'auroc':roc_auc_score(y,p),'ap':average_precision_score(y,p),'confusion_matrix':confusion_matrix(y,pred).tolist()}

def atomic_save(state,path):
    tmp=path.with_suffix('.tmp'); torch.save(state,tmp); os.replace(tmp,path)

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}

def restore_rng(d):
    random.setstate(d['python']); np.random.set_state(d['numpy']); torch.set_rng_state(d['torch']); torch.cuda.set_rng_state_all(d['cuda'])

def write_predictions(name,ds,values,threshold):
    y,p,s,z,sz=values; records=[]
    for i,r in enumerate(ds.rows):
        records.append({**r,'true_label':int(y[i]),'pred_label':int(p[i]>=threshold),'coin_score':float(p[i]),
                        'coin_logit':float(z[i]),'threshold':float(threshold)})
    write_csv(OUT/f'{name}_predictions.csv',records)
    return records

def write_csv(path,records):
    if not records: return
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)

def finish(m,config,history):
    best=torch.load(MODEL/'best.pt',map_location='cpu',weights_only=False); m.load_state_dict(best['model'])
    val=Coins('val'); values=predict(m,loader(val,config['batch_coins'],workers=0))
    precision,recall,thresholds=precision_recall_curve(values[0],values[1]); fs=2*precision[:-1]*recall[:-1]/(precision[:-1]+recall[:-1]+1e-12)
    threshold=float(thresholds[np.flatnonzero(fs==fs.max())[-1]])
    save_json(MODEL/'threshold.json',{'threshold':threshold,'source':'validation only','best_epoch':best['epoch'],'val_metrics':metrics(values[0],values[1],threshold)})
    write_predictions('val',val,values,threshold)
    save_json(OUT/'TRAIN_COMPLETE.json',dict(epochs=len(history),best_epoch=best['epoch'],training_seconds=sum(h['epoch_total_seconds'] for h in history),peak_allocated_MiB=max(h['allocated_MiB'] for h in history)))
    import evaluate
    evaluate.main()


def train():
    config=json.loads((MODEL/'config.json').read_text(encoding='utf-8'))
    if 'training_start_unix_estimate' not in config:
        config['training_start_unix_estimate']=time.time(); config['timing_start_method']='direct launch timestamp'
        save_json(MODEL/'config.json',config)
    for k,v in FILES.items(): assert hashlib.sha256((SPLIT/v).read_bytes()).hexdigest()==config['split_sha256'][k]
    seed(); m=pretrained(); opt=optimizer(m); scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    history=[]; best=-1.; stale=0; start=1; ck={}
    if (MODEL/'last.pt').exists():
        ck=torch.load(MODEL/'last.pt',map_location='cpu',weights_only=False)
        m.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        history=ck['history']; best=ck['best']; stale=ck['stale']; start=ck['epoch']+1
        print(f'RESUME epoch={start} stale={stale}',flush=True)
    schedule=config['scheduler']
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=schedule['factor'],
        patience=schedule['patience'],threshold=schedule['threshold'],threshold_mode='abs',min_lr=schedule['min_lr'])
    if 'scheduler' in ck: scheduler.load_state_dict(ck['scheduler'])
    elif history: scheduler.best=best
    for h in history:
        for k,v in [('backbone_lr',2e-5),('head_lr',1e-4),('next_backbone_lr',2e-5),('next_head_lr',1e-4)]: h.setdefault(k,v)
        h.setdefault('amp_retries',0); h.setdefault('amp_final_scale',None)
    ds=Coins('train',True); val=Coins('val'); vl=loader(val,config['batch_coins'],workers=0)
    dl=loader(ds,config['batch_coins'],True,config['workers'])
    for epoch in range(start,config['max_epochs']+1):
        if epoch>config['min_epochs'] and stale>=config['patience']: break
        used_lrs=[g['lr'] for g in opt.param_groups]
        seed(42+epoch); ds.epoch[0]=epoch
        dl.sampler.generator=torch.Generator().manual_seed(42+epoch)
        torch.cuda.reset_peak_memory_stats(); t=time.perf_counter()
        r=train_epoch(m,opt,scaler,dl,config['accumulation'])
        values=predict(m,vl); vm=metrics(values[0],values[1]); r.update(epoch=epoch,**{'val_'+k:v for k,v in vm.items() if k!='confusion_matrix'})
        if vm['auroc']>best:
            best=vm['auroc']; stale=0; atomic_save({'model':m.state_dict(),'epoch':epoch,'val_auroc':best},MODEL/'best.pt')
        else: stale+=1
        if epoch>schedule['start_after_epoch']: scheduler.step(vm['auroc'])
        else: scheduler.best=best; scheduler.num_bad_epochs=0
        r.update(backbone_lr=used_lrs[0],head_lr=used_lrs[1],next_backbone_lr=opt.param_groups[0]['lr'],next_head_lr=opt.param_groups[1]['lr'])
        r['epoch_total_seconds']=time.perf_counter()-t; history.append(r)
        atomic_save({'model':m.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'history':history,'best':best,'stale':stale,'rng':rng_state()},MODEL/'last.pt')
        r['epoch_total_seconds']=time.perf_counter()-t
        write_csv(OUT/'history.csv',history); save_json(OUT/'history.json',history)
        print('EPOCH '+json.dumps(r)+f" patience={stale}/{config['patience']}",flush=True)
        if epoch==1: print('ESTIMATE_MINUTES '+json.dumps({str(n):r['epoch_total_seconds']*n/60 for n in [10,20,30]}),flush=True)
    finish(m,config,history)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=['train'],required=True); args=parser.parse_args()
    train()



