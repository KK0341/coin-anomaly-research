"""A0: shared ConvNeXt-Tiny, max side logits, coin-level BCE. Run --mode bench/train."""
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
FILES={'train':'训练集.csv','val':'验证集.csv','test':'测试集.csv'}
torch.set_num_threads(4)

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
        # No crop, blur, color jitter or erasing: preserve surface evidence.
        ops=[T.Resize((384,384),antialias=True)]
        if augment: ops += [T.RandomRotation(10,interpolation=T.InterpolationMode.BILINEAR)]
        self.transform=T.Compose(ops+[T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; images=[]
        for side in ['正面路径','背面路径']:
            with Image.open(r[side]) as im: images.append(self.transform(im.convert('RGB')))
        return torch.stack(images),float(r['二分类标签']),i

def worker_init(worker):
    s=torch.initial_seed()%2**32; random.seed(s); np.random.seed(s); torch.set_num_threads(1)

def loader(ds,batch,shuffle=False,workers=2):
    return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=workers,pin_memory=True,
                      persistent_workers=workers>0,worker_init_fn=worker_init,
                      **({'prefetch_factor':2} if workers else {}))

class Dual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=convnext_tiny(weights=None)
        self.net.classifier[2]=nn.Linear(768,1)
    def forward(self,x):
        sides=self.net(x.flatten(0,1)).reshape(-1,2)
        return sides.max(dim=1).values,sides

def pretrained():
    p=MODEL/'convnext_tiny-983f1562.pth'
    if not p.exists():
        print('Downloading required ImageNet weights; failure is fatal (no random-init fallback).',flush=True)
        torch.hub.download_url_to_file(ConvNeXt_Tiny_Weights.IMAGENET1K_V1.url,str(p),hash_prefix='983f1562',progress=False)
    state=torch.load(p,map_location='cpu',weights_only=True)
    m=Dual(); del state['classifier.2.weight']; del state['classifier.2.bias']
    missing=m.net.load_state_dict(state,strict=False)
    assert set(missing.missing_keys)=={'classifier.2.weight','classifier.2.bias'} and not missing.unexpected_keys
    return m.cuda()

def optimizer(m):
    head=list(m.net.classifier.parameters()); ids={id(p) for p in head}
    return torch.optim.AdamW([{'params':[p for p in m.parameters() if id(p) not in ids],'lr':2e-5},
                              {'params':head,'lr':1e-4}],weight_decay=.01)

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

def benchmark():
    seed(); ds=Coins('train',True); results=[]
    for b in [4,8,16]:
        if b==16 and (not results[-1].get('ok') or results[-1]['reserved_MiB']>5500): break
        m=opt=scaler=dl=None
        try:
            seed(); m=pretrained(); opt=optimizer(m); scaler=torch.amp.GradScaler('cuda',init_scale=1024)
            dl=loader(ds,b,True); torch.cuda.reset_peak_memory_stats()
            train_epoch(m,opt,scaler,dl,limit=3)
            torch.cuda.reset_peak_memory_stats(); r=train_epoch(m,opt,scaler,dl,limit=12)
            r.update(batch=b,images=b*2,ok=True); results.append(r)
            print('BENCH '+json.dumps(r),flush=True)
        except torch.OutOfMemoryError:
            results.append({'batch':b,'ok':False,'reason':'CUDA OOM'}); print(f'BATCH {b} OOM',flush=True)
        finally:
            del m,opt,scaler,dl; gc.collect(); torch.cuda.empty_cache()
        save_json(OUT/'batch_benchmark.json',results)
    safe=[r for r in results if r['ok'] and r['reserved_MiB']<7000]
    if not safe: raise RuntimeError('No stable batch configuration')
    # Prefer requested effective batch 8; 16 is capacity exploration only.
    b=8 if any(r['batch']==8 for r in safe) else 4
    config={'batch_coins':b,'images_per_batch':2*b,'accumulation':8//b,'effective_batch_coins':8,
            'seed':42,'resolution':384,'max_epochs':50,'min_epochs':15,'patience':10,'workers':2,
            'scheduler':{'name':'ReduceLROnPlateau','mode':'max','factor':0.5,'patience':4,'threshold':1e-4,'threshold_mode':'abs','min_lr':[2e-7,1e-6],'start_after_epoch':5},
            'backbone_lr':2e-5,'head_lr':1e-4,'weight_decay':.01,'gradient_clip_norm':5,
            'weights':'ConvNeXt_Tiny_Weights.IMAGENET1K_V1','python':sys.executable,
            'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0),'total_MiB':torch.cuda.get_device_properties(0).total_memory/2**20,
            'selection':'validation AUROC','threshold':'max validation F1, highest threshold breaks ties',
            'augmentation':'resize 384x384; independent rotation +/-10 degrees; ImageNet normalization',
            'split_sha256':{k:hashlib.sha256((SPLIT/v).read_bytes()).hexdigest() for k,v in FILES.items()}}
    save_json(MODEL/'config.json',config)

def write_predictions(name,ds,values,threshold):
    y,p,s,z,sz=values; records=[]
    for i,r in enumerate(ds.rows):
        records.append({**r,'true_label':int(y[i]),'pred_label':int(p[i]>=threshold),'coin_score':float(p[i]),
                        'front_score':float(s[i,0]),'back_score':float(s[i,1]),'coin_logit':float(z[i]),
                        'front_logit':float(sz[i,0]),'back_logit':float(sz[i,1]),'threshold':float(threshold)})
    write_csv(OUT/f'{name}_predictions.csv',records)
    return records

def write_csv(path,records):
    if not records: return
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0])); w.writeheader(); w.writerows(records)

def finish(m,config,history):
    best=torch.load(MODEL/'best.pt',map_location='cpu',weights_only=False); m.load_state_dict(best['model'])
    val=Coins('val'); values=predict(m,loader(val,config['batch_coins']))
    precision,recall,thresholds=precision_recall_curve(values[0],values[1]); fs=2*precision[:-1]*recall[:-1]/(precision[:-1]+recall[:-1]+1e-12)
    threshold=float(thresholds[np.flatnonzero(fs==fs.max())[-1]])
    save_json(MODEL/'threshold.json',{'threshold':threshold,'source':'validation only','best_epoch':best['epoch'],'val_metrics':metrics(values[0],values[1],threshold)})
    write_predictions('val',val,values,threshold)
    # Persist raw test predictions immediately; a report-only restart does not re-evaluate test.
    test=Coins('test'); pred_path=OUT/'test_predictions.csv'
    if pred_path.exists():
        with open(pred_path,encoding='utf-8-sig') as f: records=list(csv.DictReader(f))
        y=np.array([int(r['true_label']) for r in records]); p=np.array([float(r['coin_score']) for r in records])
    else:
        values=predict(m,loader(test,config['batch_coins'])); records=write_predictions('test',test,values,threshold); y,p=values[:2]
    result=metrics(y,p,threshold); result.update(threshold=threshold,best_epoch=best['epoch'],epochs_completed=len(history))
    groups=[]
    for g in ['AU','XF','MS','92','97','98']:
        subset=[r for r in records if (str(r['原始评级']).startswith(g) and int(r['true_label'])==0) if g in ['AU','XF','MS']] if g in ['AU','XF','MS'] else [r for r in records if g in str(r['异常代码']) and int(r['true_label'])==1]
        errors=sum(int(r['true_label'])!=int(r['pred_label']) for r in subset)
        groups.append({'type':g,'total':len(subset),'errors':errors,'error_rate':errors/len(subset) if subset else None,'error_kind':'FP' if g in ['AU','XF','MS'] else 'FN'})
    result['groups']=groups; result['training_seconds']=sum(h['epoch_total_seconds'] for h in history)
    result['wall_elapsed_seconds']=time.time()-config.get('training_start_unix_estimate',time.time())
    result['successful_epoch_amp_retries']=sum(h.get('amp_retries',0) for h in history)
    result['peak_allocated_MiB']=max(h['allocated_MiB'] for h in history); result['peak_reserved_MiB']=max(h['reserved_MiB'] for h in history)
    save_json(OUT/'test_metrics.json',result); write_csv(OUT/'error_groups.csv',groups)
    errors=[r for r in records if int(r['true_label'])!=int(r['pred_label'])]
    examples=[]
    for label in [0,1]:
        subset=[r for r in errors if int(r['true_label'])==label]
        examples+=sorted(subset,key=lambda r:float(r['coin_score']),reverse=label==0)[:15]
    write_csv(OUT/'typical_errors.csv',examples)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4)); epochs=[h['epoch'] for h in history]
    ax[0].plot(epochs,[h['loss'] for h in history]); ax[0].set(xlabel='Epoch',ylabel='Train BCE loss')
    for k in ['auroc','f1']: ax[1].plot(epochs,[h['val_'+k] for h in history],label=k)
    ax[1].legend(); ax[1].set(xlabel='Epoch',ylabel='Validation (F1 at 0.5)'); fig.tight_layout(); fig.savefig(OUT/'training_curves.png',dpi=160); plt.close(fig)
    cm=np.array(result['confusion_matrix']); fig,ax=plt.subplots(figsize=(4,4)); ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='True',title='Test confusion matrix'); fig.tight_layout(); fig.savefig(OUT/'confusion_matrix.png',dpi=160); plt.close(fig)
    avg=result['training_seconds']/len(history)
    report=f'''# A0 双面监督二分类实验\n\n共享 ImageNet ConvNeXt-Tiny 和分类头；整币 logit=max(正面,背面)，整币 BCEWithLogitsLoss。全参数微调。单面分数为弱监督证据，不是独立校准的单面异常概率。\n\n## 环境与配置\n\n```json\n{json.dumps(config,ensure_ascii=False,indent=2)}\n```\n\n固定划分 train=6033、val=755、test=755；未修改原图和划分。测试仅在最佳模型及验证阈值确定后评估。无类别加权或额外类别信息。第 1–5 轮保持初始学习率；按用户中途确认，从第 6 轮开始启用 ReduceLROnPlateau（mode=max、factor=0.5、patience=4、absolute threshold=1e-4、最小学习率 2e-7/1e-6）。学习率下降不重置早停计数。AMP float16；各 epoch 以 42+epoch 重置随机种子，支持按 epoch 恢复。\n\n## 测速与训练\n\n首轮真实训练和验证耗时 {history[0]['epoch_total_seconds']:.1f} 秒；平均 {avg:.1f} 秒/轮；累计 {result['training_seconds']/60:.1f} 分钟（训练、验证、checkpoint 写入；不含环境检查、下载、独立 batch 测速、未完成轮次及恢复间隔）。\n首轮训练速度 {history[0]['coins_per_second']:.2f} 枚/秒，数据等待占比 {history[0]['data_wait_fraction']:.1%}。\n基于首轮的 10/15/20/30/50 轮预计分钟数：{[round(history[0]['epoch_total_seconds']*n/60,1) for n in [10,15,20,30,50]]}。\n峰值分配/保留显存 {result['peak_allocated_MiB']:.0f}/{result['peak_reserved_MiB']:.0f} MiB（PyTorch allocator；驱动显示占用可能更高）。\n最佳 epoch={best['epoch']}，实际完成 {len(history)} 轮，early stopping patience={config['patience']}，至少 {config['min_epochs']} 轮、最多 {config['max_epochs']} 轮。每轮 F1 使用固定 0.5，仅最终阈值使用验证集最大 F1（并列选最高阈值）。\n\n总墙钟耗时（含配置切换、未完成轮次、恢复和最终评估）约 {result['wall_elapsed_seconds']/3600:.2f} 小时；起点依据见 config.timing_start_method。第 7 轮首次尝试曾因 AMP 非有限梯度安全中止，恢复自第 6 轮；已修复为缩小 scale 并重试同一批次，溢出时不更新参数，不跳过样本。恢复记录见结果/amp_recovery.json。\n\n## 最终测试\n\n```json\n{json.dumps(result,ensure_ascii=False,indent=2)}\n```\n\n![曲线](结果/training_curves.png)\n\n![混淆矩阵](结果/confusion_matrix.png)\n\n错误分组详见结果/error_groups.csv，典型高置信度错误详见结果/typical_errors.csv；完整双面与整币 logit/score 详见结果/test_predictions.csv。AU/XF/MS 为正常币误报，92/97/98 为异常币漏检；这些子组不代表全部评级。\n\n## 复现与恢复\n\n使用 config.json 中 python 运行 `代码/train.py --mode train`，自动从模型/last.pt 恢复；最佳权重模型/best.pt。先运行 `--mode bench` 可重测 batch（不要覆盖已完成实验配置）。独立测速模型随后丢弃，正式训练重新加载 ImageNet 权重。\n'''
    (ROOT/'实验报告.md').write_text(report,encoding='utf-8'); print('FINAL '+json.dumps(result,ensure_ascii=False),flush=True)

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
    ds=Coins('train',True); val=Coins('val'); vl=loader(val,config['batch_coins'])
    for epoch in range(start,config['max_epochs']+1):
        if epoch>config['min_epochs'] and stale>=config['patience']: break
        used_lrs=[g['lr'] for g in opt.param_groups]
        seed(42+epoch); dl=loader(ds,config['batch_coins'],True,config['workers'])
        torch.cuda.reset_peak_memory_stats(); t=time.perf_counter()
        r=train_epoch(m,opt,scaler,dl,config['accumulation']); del dl
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
    parser=argparse.ArgumentParser(); parser.add_argument('--mode',choices=['bench','train'],required=True); args=parser.parse_args()
    if args.mode=='bench': benchmark()
    else: train()
