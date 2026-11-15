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
ROOT=Path(__file__).resolve().parents[2]; CODES=['92','97','98','95','94','91']; NAMES=['清洗','环境损伤','损坏','划痕','表面改动','颜色存疑']
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
        g=torch.Generator().manual_seed(42004)
        self.centers=nn.Parameter(nn.functional.normalize(torch.randn(6,256,generator=g),dim=1))
    def forward(self,x,features=False):
        h=self.head[1](self.head[0](self.net(x.flatten(0,1)).reshape(-1,1536)))
        z=self.head[2](h)
        return (z,h) if features else z
    def center_distances(self,h,y):
        return (nn.functional.normalize(h.float(),dim=1)-nn.functional.normalize(self.centers.float(),dim=1)[y]).square().sum(1)
def loader(ds,shuffle=False):
    return DataLoader(ds,batch_size=4,shuffle=shuffle,num_workers=4,pin_memory=True,generator=torch.Generator().manual_seed(42+ds.epoch))
def metrics(y,p):
    pred=p.argmax(1); a,b,c,n=precision_recall_fscore_support(y,pred,labels=range(6),zero_division=0)
    return dict(accuracy=accuracy_score(y,pred),macro_f1=f1_score(y,pred,labels=range(6),average='macro',zero_division=0),balanced_accuracy=balanced_accuracy_score(y,pred),per_class={k:dict(name=NAMES[i],precision=float(a[i]),recall=float(b[i]),f1=float(c[i]),support=int(n[i])) for i,k in enumerate(CODES)},confusion_matrix=confusion_matrix(y,pred,labels=range(6)).tolist())
@torch.inference_mode()
def predict(m,ds,w):
    m.eval(); ys=[]; ps=[]; total=0.; denom=0.; center_eval_sum=0.; center_eval_n=0
    for x,y,_ in loader(ds):
        with torch.autocast('cuda',dtype=torch.bfloat16): z,h=m(x.cuda(non_blocking=True),features=True)
        center_eval_sum+=m.center_distances(h,y.cuda()).sum().item(); center_eval_n+=len(y)
        z=z.float(); keep=y>=0
        if keep.any():
            yy=y[keep].cuda(); total+=nn.functional.cross_entropy(z[keep],yy,weight=w,reduction='sum').item(); denom+=w[yy].sum().item()
        ys.extend(y.tolist()); ps.extend(z.softmax(1).cpu().tolist())
    m.last_val_center_loss=center_eval_sum/center_eval_n
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
    save(out/'validation_selection.json',dict(best_epoch=torch.load(out/'best.pt',map_location='cpu',weights_only=False)['epoch'],test_policy='checkpoint and configuration frozen before one final test'))
    from evaluate import finish_all
    finish_all(m,out,manifest,w,history,cfg)

def run(name):
    out=ROOT/name; out.mkdir(exist_ok=True)
    if (out/'COMPLETE.json').exists(): return
    manifest=json.loads((DATA/'manifest.json').read_text('utf8')); trainrows=[r for r in manifest['train'] if r['异常代码'] in CODES and r['二分类标签']=='1']; valrows=[r for r in manifest['val'] if r['异常代码'] in CODES and r['二分类标签']=='1']
    seed(42); m=Dual(); initial_head={k:v.clone() for k,v in m.head[2].state_dict().items()}
    if name=='B4':
        state=torch.load(DATA/'a8.pt',map_location='cpu',weights_only=False)['model']; state.pop('head.2.weight'); state.pop('head.2.bias'); missing=m.load_state_dict(state,strict=False); assert set(missing.missing_keys)=={'head.2.weight','head.2.bias','centers'} and not missing.unexpected_keys
    else:
        state=torch.load(DATA/'imagenet.pth',map_location='cpu',weights_only=True); state.pop('classifier.2.weight'); state.pop('classifier.2.bias'); m.net.load_state_dict(state,strict=True)
    assert all(torch.equal(v,initial_head[k]) for k,v in m.head[2].state_dict().items())
    m.cuda(); opt=torch.optim.AdamW([{'params':m.net.parameters(),'lr':1e-5},{'params':m.head.parameters(),'lr':1e-4},{'params':[m.centers],'lr':1e-4,'weight_decay':0.}],weight_decay=.01)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=4,threshold=0,threshold_mode='abs',min_lr=[1e-7,1e-6,1e-6])
    counts=np.bincount([CODES.index(r['异常代码']) for r in trainrows],minlength=6); weights=1/np.sqrt(counts); weights/=weights.mean(); w=torch.tensor(weights,dtype=torch.float32,device='cuda')
    cfg={'experiment':name,'initialization':'A8 binary backbone + hidden MLP; new six-class output' if name=='B4' else 'ImageNet backbone + seeded random MLP','classes':dict(zip(CODES,NAMES)),'class_counts':counts.tolist(),'class_weights':weights.tolist(),'seed':42,'resolution':512,'batch':4,'accumulation':2,'effective_batch':8,'backbone_lr':1e-5,'head_lr':1e-4,'max_epochs':60,'patience':10,'selection':'val Macro-F1; earliest epoch breaks ties','scheduler':'ReduceLROnPlateau factor .5 patience4','amp':'bfloat16','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'data_audit':json.loads((DATA/'data_audit.json').read_text('utf8')),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}; save(out/'config.json',cfg)
    cfg.update(center_weight=.1,center_location='post-GELU 256-dimensional hidden features',center_loss='mean squared Euclidean distance of L2 normalized features and target centers; float32',center_optimizer='AdamW lr1e-4 weight_decay0; same plateau schedule min_lr1e-6',center_initialization='independent generator seed42004, unit random centers; does not alter B0 network RNG',weight_decay=.01,workers=4,gradient_clip_norm=5,execution_host='local Windows only',data_directory=str(DATA.resolve()),output_directory=str(out.resolve()))
    save(out/'config.json',cfg)
    print(f'{name} START gpu={torch.cuda.get_device_name()} train={len(trainrows)} val={len(valrows)} batch=4 accumulation=2',flush=True)
    history=[]; best=-1.; stale=0; start=1
    if (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False); m.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler']); history=ck['history']; best=ck['best']; stale=ck['stale']; start=ck['epoch']+1
    ds=Coins(trainrows,True); val=Coins(valrows); initial_centers=nn.functional.normalize(m.centers.detach(),dim=1).clone()
    assert len(trainrows)==1583 and len(valrows)==198
    assert '5060' in torch.cuda.get_device_name()
    for epoch in range(start,61):
        if stale>=10: break
        seed(42+epoch); ds.epoch=epoch; dl=loader(ds,True); m.train(); t=time.time(); total=0.; denom=0.; yy=[]; pp=[]; it=iter(dl); center_sum=0.; center_counts=np.zeros(6); center_by_class=np.zeros(6); grad_by_class=np.zeros(6); before_centers=m.centers.detach().clone()
        for offset in range(0,len(dl),2):
            batches=[next(it) for _ in range(min(2,len(dl)-offset))]; group_den=sum(w[y.cuda()].sum().item() for _,y,_ in batches); group_n=sum(len(y) for _,y,_ in batches); opt.zero_grad(set_to_none=True)
            for x,y,_ in batches:
                y=y.cuda(non_blocking=True)
                with torch.autocast('cuda',dtype=torch.bfloat16): z,hfeat=m(x.cuda(non_blocking=True),features=True)
                cl=m.center_distances(hfeat,y); center_sum+=cl.detach().sum().item()
                for k in range(6):
                    mask=y==k; center_counts[k]+=mask.sum().item(); center_by_class[k]+=cl.detach()[mask].sum().item()
                loss=nn.functional.cross_entropy(z.float(),y,weight=w,reduction='sum'); assert torch.isfinite(loss); (loss/group_den+.1*cl.sum()/group_n).backward(); total+=loss.item(); denom+=w[y].sum().item(); yy.extend(y.cpu().tolist()); pp.extend(z.detach().float().softmax(1).cpu().tolist())
            assert m.centers.grad is not None and torch.isfinite(m.centers.grad).all()
            grad_by_class+=m.centers.grad.detach().norm(dim=1).cpu().numpy()
            nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True); opt.step()
            if offset==0 or (offset+2)%80==0:
                print(f'{name} epoch={epoch} batches={min(offset+2,len(dl))}/{len(dl)} loss={total/denom:.6f} seconds={time.time()-t:.1f}',flush=True)
        vy,vp,vl=predict(m,val,w); f=metrics(vy,vp)['macro_f1']; improved=f>best
        if improved: best=f; stale=0; atomic({'model':m.state_dict(),'epoch':epoch,'val_macro_f1':f,'config':cfg},out/'best.pt')
        else: stale+=1
        sched.step(f); h={'epoch':epoch,'train_loss':total/denom,'val_loss':vl,'train_macro_f1':metrics(np.array(yy),np.array(pp))['macro_f1'],'val_macro_f1':f,'seconds':time.time()-t,'stale':stale,'backbone_lr':opt.param_groups[0]['lr']}; h.update(val_center_loss=m.last_val_center_loss,val_total_loss=vl+.1*m.last_val_center_loss,train_ce=total/denom,train_center_loss=center_sum/len(ds),train_total_loss=total/denom+.1*center_sum/len(ds),center_lr=opt.param_groups[2]['lr'],peak_memory_MiB=torch.cuda.max_memory_allocated()/2**20); history.append(h)
        cn=nn.functional.normalize(m.centers.detach(),dim=1)
        center_log=dict(epoch=epoch,codes=CODES,class_counts=center_counts.tolist(),class_center_loss=(center_by_class/center_counts).tolist(),summed_gradient_norm=grad_by_class.tolist(),parameter_update_norm=(m.centers.detach()-before_centers).norm(dim=1).cpu().tolist(),center_norms=m.centers.detach().norm(dim=1).cpu().tolist(),cosine_similarity=(cn@cn.T).cpu().tolist(),distance_from_initial=(cn-initial_centers).norm(dim=1).cpu().tolist())
        with (out/'center_history.jsonl').open('a',encoding='utf8') as cf: cf.write(json.dumps(center_log)+'\n')
        atomic({'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':epoch,'history':history,'best':best,'stale':stale},out/'last.pt'); save(out/'history.json',history); csvwrite(out/'history.csv',history); print(name,json.dumps(h),flush=True)
    finish(m,out,manifest,w,history,cfg)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('experiment',choices=['B4']); args=p.parse_args(); run(args.experiment)
