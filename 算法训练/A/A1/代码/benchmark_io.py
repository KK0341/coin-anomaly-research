import train as t
from pathlib import Path
import time,json,hashlib,shutil,io,gc,random
from PIL import Image
import torch
from torch.utils.data import Subset
CACHE=Path(r'D:\coin_image_cache\dataset_v1_original')
def profile(paths):
    out=[]
    transform=t.Coins('train',True).transform
    for p in paths:
        a=time.perf_counter(); data=Path(p).read_bytes(); b=time.perf_counter()
        with Image.open(io.BytesIO(data)) as im: rgb=im.convert('RGB')
        c=time.perf_counter(); transform(rgb); d=time.perf_counter()
        out.append([b-a,c-b,d-c,len(data)])
    return dict(zip(['read_seconds','decode_seconds','transform_seconds','bytes'],[sum(x[i] for x in out) for i in range(4)]))
def bench(cfg,manifest=None,steps=32):
    t.LOADER_OPTIONS=cfg.copy(); t.CACHE_MAP=manifest or {}
    t.seed(); ds=t.Coins('train',True)
    ids=random.Random(42).sample(range(len(ds)),(steps+4)*8)
    dl=t.loader(Subset(ds,ids),8,False)
    m=t.pretrained(); opt=t.optimizer(m); scaler=torch.amp.GradScaler('cuda',init_scale=1024)
    a=time.perf_counter(); t.train_epoch(m,opt,scaler,dl,limit=4); startup=time.perf_counter()-a
    torch.cuda.reset_peak_memory_stats(); r=t.train_epoch(m,opt,scaler,dl,limit=steps)
    r.update(cfg=cfg,source='SSD' if manifest else 'E HDD',startup_warmup_seconds=startup)
    print('BENCH '+json.dumps(r),flush=True)
    del dl,m,opt,scaler; gc.collect(); torch.cuda.empty_cache()
    return r

def main():
    paths=list(dict.fromkeys(r[s] for split in t.FILES for r in t.rows(split) for s in ['正面路径','背面路径']))
    sample=random.Random(42).sample(paths,96)
    result={'profile_original':profile(sample),'notes':'OS cache not flushed; fixed sampled records, 4 warmup + 32 training steps; includes forward/backward/AdamW. No A0 inference.'}
    t.save_json(t.OUT/'io_benchmark.json',result); print('PROFILE '+json.dumps(result),flush=True)
    baseline=bench(dict(workers=2,pin_memory=True,persistent_workers=True,prefetch_factor=2)); result['benchmarks']=[baseline]
    sizes=[Path(p).stat().st_size for p in paths]; total=sum(sizes); free=shutil.disk_usage('D:\\').free
    assert free>total*1.1+20*2**30, (free,total)
    print(f'CACHE files={len(paths)} bytes={total} free={free}',flush=True)
    CACHE.mkdir(parents=True,exist_ok=True); manifest={}; audit=[]; start=time.perf_counter()
    for i,p in enumerate(paths):
        dst=CACHE/(hashlib.sha256(p.encode()).hexdigest()+Path(p).suffix)
        src=Path(p); data=src.read_bytes(); sha=hashlib.sha256(data).hexdigest()
        if not dst.exists(): dst.write_bytes(data)
        assert hashlib.sha256(dst.read_bytes()).hexdigest()==sha
        manifest[p]=str(dst); audit.append({'source':p,'cache':str(dst),'bytes':len(data),'sha256':sha})
        if (i+1)%1000==0: print(f'CACHE {i+1}/{len(paths)} {time.perf_counter()-start:.1f}s',flush=True)
    t.save_json(CACHE/'manifest.json',manifest); t.save_json(CACHE/'audit.json',audit)
    result['cache']={'files':len(paths),'bytes':total,'free_before':free,'seconds':time.perf_counter()-start,'manifest':str(CACHE/'manifest.json'),'verification':'all original/cache SHA256 equal'}
    result['profile_ssd']=profile([manifest[p] for p in sample])
    for workers,pin,persist,prefetch in [(0,True,False,2),(2,True,True,2),(4,True,True,2),(6,True,True,2),(4,False,True,2),(4,True,False,2),(4,True,True,4)]:
        cfg=dict(workers=workers,pin_memory=pin,persistent_workers=persist,prefetch_factor=prefetch)
        result['benchmarks'].append(bench(cfg,manifest)); t.save_json(t.OUT/'io_benchmark.json',result)
    best=min(result['benchmarks'][1:],key=lambda r:r['seconds'])
    chosen={**best['cfg'],'cache_manifest':str(CACHE/'manifest.json')}
    t.save_json(t.OUT/'loader_selection.json',chosen)
    c=json.loads((t.MODEL/'config.json').read_text(encoding='utf-8')); c.update(workers=chosen['workers'],loader=chosen); t.save_json(t.MODEL/'config.json',c)
    result['selected']=chosen; result['estimated_train_epoch_seconds']=6033/best['coins_per_second']; result['speedup']=best['coins_per_second']/baseline['coins_per_second']
    t.save_json(t.OUT/'io_benchmark.json',result)
    print('DONE '+json.dumps({k:v for k,v in result.items() if k not in ['benchmarks']}),flush=True)
if __name__=='__main__': main()
