import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,csv,random,time,hashlib,argparse
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.models import convnext_tiny
from torchvision import transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,precision_recall_fscore_support,confusion_matrix
ROOT=Path(__file__).resolve().parent.parent; CODES=['92','97','98','95','94','91']; NAMES=['清洗','环境损伤','损坏','划痕','表面改动','颜色存疑']
DATA=Path(os.environ.get('COIN_DATA_DIR',str(ROOT/'B0/数据缓存')))
torch.set_num_threads(4)
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
def atomic(x,p):
    torch.save(x,p.with_suffix('.tmp')); os.replace(p.with_suffix('.tmp'),p)
def csvwrite(p,rs):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0])); w.writeheader(); w.writerows(rs)
class Coins(Dataset):
    def __init__(self,rows,augment=False):
        self.rows=rows; self.augment=augment; self.epoch=0
        self.transform=T.Compose(([T.RandomRotation(10,interpolation=T.InterpolationMode.BILINEAR)] if augment else [])+[T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        if self.augment: torch.manual_seed(42+self.epoch*100000+i)
        r=self.rows[i]; ims=[]
        for p in r['images']:
            with Image.open(DATA/p) as im: ims.append(self.transform(im.convert('RGB')))
        return torch.stack(ims),CODES.index(r['异常代码']) if r['二分类标签']=='1' and r['异常代码'] in CODES else -1,i
class Dual(nn.Module):
    def __init__(self):
        super().__init__(); self.net=convnext_tiny(weights=None); self.net.classifier[2]=nn.Identity()
        self.head=nn.Sequential(nn.Linear(1536,256),nn.GELU(),nn.Linear(256,6))
    def forward(self,x): return self.head(self.net(x.flatten(0,1)).reshape(-1,1536))
def loader(ds,shuffle=False):
    return DataLoader(ds,batch_size=4,shuffle=shuffle,num_workers=4,pin_memory=True,generator=torch.Generator().manual_seed(42+ds.epoch))
def metrics(y,p):
    pred=p.argmax(1); a,b,c,n=precision_recall_fscore_support(y,pred,labels=range(6),zero_division=0)
    return dict(accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,labels=range(6),average='macro',zero_division=0),balanced_accuracy=balanced_accuracy_score(y,pred),per_class={k:dict(name=NAMES[i],precision=float(a[i]),recall=float(b[i]),f1=float(c[i]),support=int(n[i])) for i,k in enumerate(CODES)},confusion_matrix=confusion_matrix(y,pred,labels=range(6)).tolist())
@torch.inference_mode()
def predict(m,ds,w):
    m.eval(); ys=[]; ps=[]; total=0.; denom=0.
    for x,y,_ in loader(ds):
        with torch.autocast('cuda',dtype=torch.bfloat16): z=m(x.cuda(non_blocking=True))
        z=z.float(); keep=y>=0
        if keep.any():
            yy=y[keep].cuda(); total+=nn.functional.cross_entropy(z[keep],yy,weight=w,reduction='sum').item(); denom+=w[yy].sum().item()
        ys.extend(y.tolist()); ps.extend(z.softmax(1).cpu().tolist())
    return np.array(ys),np.array(ps),total/denom if denom else 0.
def plot_finish(out,ds,y,p,split):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname=str(DATA/'font.ttf'))
    pred=p.argmax(1); cm=confusion_matrix(y,pred,labels=range(6))
    for norm in [False,True]:
        a=cm/cm.sum(1,keepdims=True) if norm else cm
        fig,ax=plt.subplots(figsize=(8,7)); ax.imshow(a,cmap='Blues')
        for i in range(6):
            for j in range(6): ax.text(j,i,f'{a[i,j]:.2f}' if norm else str(a[i,j]),ha='center',va='center')
        ax.set(xticks=range(6),yticks=range(6),xticklabels=CODES,yticklabels=CODES,xlabel='Predicted',ylabel='True'); fig.tight_layout(); fig.savefig(out/f'{split}_confusion_{"normalized" if norm else "counts"}.png',dpi=150); plt.close(fig)
        np.savetxt(out/f'{split}_confusion_{"normalized" if norm else "counts"}.csv',a,delimiter=',',header=','.join(CODES),comments='')
    errors=np.flatnonzero(y!=pred); folder=out/f'{split}_errors'; folder.mkdir(exist_ok=True)
    for start in range(0,len(errors),8):
        subset=errors[start:start+8]; fig,axs=plt.subplots(len(subset),2,figsize=(8,4*len(subset)),squeeze=False)
        for j,i in enumerate(subset):
            r=ds.rows[i]
            for side in range(2):
                with Image.open(DATA/r['images'][side]) as im: axs[j,side].imshow(im)
                axs[j,side].axis('off')
            axs[j,0].set_title(f"{r['钱币编号']} 真实：{NAMES[y[i]]}",fontproperties=font)
            axs[j,1].set_title(f'预测：{NAMES[pred[i]]} 置信度 {p[i,pred[i]]:.3f}',fontproperties=font)
        fig.tight_layout(); fig.savefig(folder/f'errors_{start//8+1:03}.jpg',dpi=100); plt.close(fig)
def finish(m,out,manifest,w,history,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    best=torch.load(out/'best.pt',map_location='cpu',weights_only=False); m.load_state_dict(best['model']); results={}
    for split in ['val','test']:
        ds=Coins([r for r in manifest[split] if r['二分类标签']=='1' and r['异常代码'] in CODES]); cached=out/f'{split}_final_inference.npz'
        if cached.exists():
            values=np.load(cached); y=values['y']; p=values['p']; loss=float(values['loss']); assert int(values['epoch'])==best['epoch']
        else:
            y,p,loss=predict(m,ds,w); np.savez_compressed(cached,y=y,p=p,loss=loss,epoch=best['epoch'])
        mask=y>=0
        six=Coins([r for r,k in zip(ds.rows,mask) if k]); yy=y[mask]; pp=p[mask]
        result=metrics(yy,pp); result['loss']=loss; results[split]=result
        rs=[]
        for i,r in enumerate(six.rows):
            pr=int(pp[i].argmax()); rs.append({**{k:v for k,v in r.items() if k!='images'},'true_class':CODES[yy[i]],'pred_class':CODES[pr],'confidence':float(pp[i,pr]),**{f'p_{c}':float(pp[i,j]) for j,c in enumerate(CODES)}})
        csvwrite(out/f'{split}_predictions.csv',rs); save(out/f'{split}_metrics.json',result); plot_finish(out,six,yy,pp,split)
    fig,ax=plt.subplots(1,2,figsize=(12,4)); ep=[h['epoch'] for h in history]
    for key in ['train_loss','val_loss']: ax[0].plot(ep,[h[key] for h in history],label=key)
    for key in ['train_macro_f1','val_macro_f1']: ax[1].plot(ep,[h[key] for h in history],label=key)
    for a in ax: a.legend(); a.set_xlabel('Epoch')
    fig.tight_layout(); fig.savefig(out/'training_curves.png',dpi=160); plt.close(fig)
    r=results['test']; lines=[f'# {out.name} 实验报告',f'初始化：{cfg["initialization"]}。最佳轮次 {best["epoch"]}，共训练 {len(history)} 轮。',f'独立六分类测试 Accuracy={r["accuracy"]:.6f}，Macro-F1={r["macro_f1"]:.6f}，Balanced Accuracy={r["balanced_accuracy"]:.6f}。','', '|原因|Precision|Recall|F1|','|---|---:|---:|---:|']
    for k,v in r['per_class'].items(): lines.append(f'|{k} {v["name"]}|{v["precision"]:.4f}|{v["recall"]:.4f}|{v["f1"]:.4f}|')
    lines+=['','固定原划分、真实异常标签筛选；82/93保留于原始manifest，排除六类指标。验证Macro-F1选模；测试仅在最终选模后推理一次。','共享ConvNeXt-Tiny → 双面拼接1536维 → Linear256/GELU/Linear6。512输入、±10°旋转、ImageNet归一化，无颜色扰动。AdamW 1e-5/1e-4、weight decay .01；batch4×累积2=8；BF16；逆平方根类别权重均值归一化；60轮上限、patience10、验证F1平台降学习率。','本轮仅评价真实异常币的六分类任务；不重新评价阶段一分流。','少数类测试数量仅8—10，单次划分结果波动较大；混淆不等于已证实的视觉成因。']
    (out/'实验报告.md').write_text('\n\n'.join(lines),'utf8'); save(out/'COMPLETE.json',{'best_epoch':best['epoch'],'epochs':len(history),'metrics':results})
from model_data import Coins, Dual, loader

def run(name):
    out=ROOT/name; out.mkdir(exist_ok=True)
    if (out/'COMPLETE.json').exists(): return
    manifest=json.loads((DATA/'manifest.json').read_text('utf8')); trainrows=[r for r in manifest['train'] if r['异常代码'] in CODES and r['二分类标签']=='1']; valrows=[r for r in manifest['val'] if r['异常代码'] in CODES and r['二分类标签']=='1']
    seed(42); m=Dual(); initial_head={k:v.clone() for k,v in m.head[2].state_dict().items()}
    if name=='B3':
        state=torch.load(DATA/'a8.pt',map_location='cpu',weights_only=False)['model']; state.pop('head.2.weight'); state.pop('head.2.bias'); missing=m.load_state_dict(state,strict=False); assert set(missing.missing_keys)=={'head.2.weight','head.2.bias','local_fuse.weight'} and not missing.unexpected_keys
    else:
        state=torch.load(DATA/'imagenet.pth',map_location='cpu',weights_only=True); state.pop('classifier.2.weight'); state.pop('classifier.2.bias'); m.net.load_state_dict(state,strict=True)
    assert all(torch.equal(v,initial_head[k]) for k,v in m.head[2].state_dict().items())
    m.cuda(); opt=torch.optim.AdamW([{'params':m.net.parameters(),'lr':1e-5},{'params':list(m.head.parameters())+list(m.local_fuse.parameters()),'lr':1e-4}],weight_decay=.01)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=4,threshold=0,threshold_mode='abs',min_lr=[1e-7,1e-6])
    counts=np.bincount([CODES.index(r['异常代码']) for r in trainrows],minlength=6); weights=1/np.sqrt(counts); weights/=weights.mean(); w=torch.tensor(weights,dtype=torch.float32,device='cuda')
    cfg={'experiment':name,'initialization':'A8 binary backbone + hidden MLP; new six-class output' if name=='B3' else 'ImageNet backbone + seeded random MLP','classes':dict(zip(CODES,NAMES)),'class_counts':counts.tolist(),'class_weights':weights.tolist(),'seed':42,'resolution':512,'batch':4,'accumulation':2,'effective_batch':8,'backbone_lr':1e-5,'head_lr':1e-4,'max_epochs':60,'patience':10,'selection':'val Macro-F1; earliest epoch breaks ties','scheduler':'ReduceLROnPlateau factor .5 patience4','amp':'bfloat16','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'data_audit':json.loads((DATA/'data_audit.json').read_text('utf8')),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}; save(out/'config.json',cfg)
    cfg.update(execution_host='local Windows only',data_directory=str(DATA.resolve()),output_directory=str(out.resolve()))
    cfg.update(method='global512 + 4 original512 quadrants/side; shared first two stages; mean/max pool; zero-init 768x256 fusion',workers=2,local_resolution=512,local_regions_per_side=4,new_loss=False)
    save(out/'config.json',cfg)
    print(f'{name} START gpu={torch.cuda.get_device_name()} train={len(trainrows)} val={len(valrows)} batch=4 accumulation=2',flush=True)
    history=[]; best=-1.; stale=0; start=1
    if (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False); m.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler']); history=ck['history']; best=ck['best']; stale=ck['stale']; start=ck['epoch']+1
    ds=Coins(trainrows,True); val=Coins(valrows)
    for epoch in range(start,61):
        if stale>=10: break
        seed(42+epoch); ds.epoch=epoch; dl=loader(ds,True); m.train(); t=time.time(); torch.cuda.reset_peak_memory_stats(); total=0.; denom=0.; yy=[]; pp=[]; it=iter(dl)
        for offset in range(0,len(dl),2):
            batches=[next(it) for _ in range(min(2,len(dl)-offset))]; group_den=sum(w[y.cuda()].sum().item() for _,y,_ in batches); opt.zero_grad(set_to_none=True)
            for x,y,_ in batches:
                y=y.cuda(non_blocking=True)
                with torch.autocast('cuda',dtype=torch.bfloat16): z=m(x.cuda(non_blocking=True))
                loss=nn.functional.cross_entropy(z.float(),y,weight=w,reduction='sum'); assert torch.isfinite(loss); (loss/group_den).backward(); total+=loss.item(); denom+=w[y].sum().item(); yy.extend(y.cpu().tolist()); pp.extend(z.detach().float().softmax(1).cpu().tolist())
            nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True); opt.step()
            if offset==0 or (offset+2)%80==0:
                print(f'{name} epoch={epoch} batches={min(offset+2,len(dl))}/{len(dl)} loss={total/denom:.6f} seconds={time.time()-t:.1f}',flush=True)
        vy,vp,vl=predict(m,val,w); f=metrics(vy,vp)['macro_f1']; improved=f>best
        if improved: best=f; stale=0; atomic({'model':m.state_dict(),'epoch':epoch,'val_macro_f1':f,'config':cfg},out/'best.pt')
        else: stale+=1
        sched.step(f); h={'epoch':epoch,'train_loss':total/denom,'val_loss':vl,'train_macro_f1':metrics(np.array(yy),np.array(pp))['macro_f1'],'val_macro_f1':f,'seconds':time.time()-t,'peak_memory_MiB':torch.cuda.max_memory_allocated()/2**20,'stale':stale,'backbone_lr':opt.param_groups[0]['lr']}; history.append(h)
        atomic({'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':epoch,'history':history,'best':best,'stale':stale},out/'last.pt'); save(out/'history.json',history); csvwrite(out/'history.csv',history); print(name,json.dumps(h),flush=True)
    save(out/'validation_selection.json',{'best_epoch':max(history,key=lambda h:h['val_macro_f1'])['epoch'],'best_val_macro_f1':best,'epochs':len(history),'test_policy':'method and checkpoint frozen before one final test; previously reused test is descriptive only'})
    finish(m,out,manifest,w,history,cfg)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('experiment',choices=['B3']); args=p.parse_args(); run(args.experiment)
