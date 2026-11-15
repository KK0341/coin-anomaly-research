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
def frozen_hash(m):
    h=hashlib.sha256()
    for name,p in m.named_parameters():
        if not p.requires_grad:
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
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
        ds=Coins(manifest[split]); cached=out/f'{split}_final_inference.npz'
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
        if out.name=='B2' and split=='val':
            baseline=json.loads((ROOT/'B0/val_metrics.json').read_text('utf8'))
            save(out/'validation_selection.json',{'best_epoch':best['epoch'],'val_macro_f1':result['macro_f1'],'val_accuracy':result['accuracy'],'delta_macro_f1_vs_B0':result['macro_f1']-baseline['macro_f1'],'effective_by_validation':result['macro_f1']>baseline['macro_f1'],'decision':'frozen model; proceed to one final test evaluation; no tuning'})
        # Reuse frozen A8 binary predictions; every missed six-class coin remains an error.
        with (DATA/f'a8_{split}.csv').open(encoding='utf-8-sig') as f: binary={r['钱币编号']:r for r in csv.DictReader(f)}
        assert set(binary)=={r['钱币编号'] for r in ds.rows}
        detected=np.array([int(binary[r['钱币编号']]['pred_label']) for r in ds.rows],dtype=bool)
        routed=np.where(detected,p.argmax(1),6); truth=np.where(np.array([r['二分类标签']=='0' for r in ds.rows]),6,y)
        covered=truth>=0; cy=truth[covered]; cp=routed[covered]
        aa,bb,cc,nnn=precision_recall_fscore_support(cy,cp,labels=range(7),zero_division=0)
        flow={'stage1':'A8 frozen best checkpoint and validation-selected threshold; not ensemble','threshold':float(next(iter(binary.values()))['threshold']),'covered_n':int(covered.sum()),'accuracy_7class_covered':accuracy_score(cy,cp),'macro_f1_7class_covered':float(cc.mean()),'six_class_f1_in_full_flow':dict(zip(CODES,cc[:6].tolist())),'six_class_macro_f1_in_full_flow':float(cc[:6].mean()),'six_abnormal_n':int(mask.sum()),'six_abnormal_stage1_missed':int((mask&~detected).sum()),'six_abnormal_end_to_end_accuracy':float((routed[mask]==y[mask]).mean()),'uncovered_n':int((~covered).sum()),'uncovered_detected':int(detected[~covered].sum()),'normal_false_positives':int(((truth==6)&detected).sum()),'confusion_labels':CODES+['normal'],'confusion_matrix':confusion_matrix(cy,cp,labels=range(7)).tolist()}
        save(out/f'{split}_pipeline_metrics.json',flow)
        csvwrite(out/f'{split}_pipeline_predictions.csv',[{'coin_id':r['钱币编号'],'original_reason':r['异常代码'],'binary_true':r['二分类标签'],'stage1_anomaly_probability':binary[r['钱币编号']]['coin_score'],'stage1_pred':int(detected[i]),'coverage':'covered' if covered[i] else 'uncovered','pipeline_pred':CODES[routed[i]] if routed[i]<6 else 'normal',**{f'p_{c}':float(p[i,j]) for j,c in enumerate(CODES)}} for i,r in enumerate(ds.rows)])
    fig,ax=plt.subplots(1,2,figsize=(12,4)); ep=[h['epoch'] for h in history]
    for key in ['train_loss','val_loss']: ax[0].plot(ep,[h[key] for h in history],label=key)
    for key in ['train_macro_f1','val_macro_f1']: ax[1].plot(ep,[h[key] for h in history],label=key)
    for a in ax: a.legend(); a.set_xlabel('Epoch')
    fig.tight_layout(); fig.savefig(out/'training_curves.png',dpi=160); plt.close(fig)
    r=results['test']; lines=[f'# {out.name} 实验报告',f'初始化：{cfg["initialization"]}。最佳轮次 {best["epoch"]}，共训练 {len(history)} 轮。',f'独立六分类测试 Accuracy={r["accuracy"]:.6f}，Macro-F1={r["macro_f1"]:.6f}，Balanced Accuracy={r["balanced_accuracy"]:.6f}。','', '|原因|Precision|Recall|F1|','|---|---:|---:|---:|']
    for k,v in r['per_class'].items(): lines.append(f'|{k} {v["name"]}|{v["precision"]:.4f}|{v["recall"]:.4f}|{v["f1"]:.4f}|')
    lines+=['','固定原划分、真实异常标签筛选；82/93保留于原始manifest，排除六类指标。验证Macro-F1选模；测试仅在最终选模后推理一次。','共享ConvNeXt-Tiny → 双面拼接1536维 → Linear256/GELU/Linear6。512输入、±10°旋转、ImageNet归一化，无颜色扰动。AdamW 1e-5/1e-4、weight decay .01；batch4×累积2=8；BF16；逆平方根类别权重均值归一化；60轮上限、patience10、验证F1平台降学习率。','pipeline文件为A8单模型接入效果，使用阶段一已冻结阈值与逐币预测，包含漏检、正常币误报；未覆盖原因单列。未运行A0+A8+A5b集成流程。','少数类测试数量仅8—10，单次划分结果波动较大；混淆不等于已证实的视觉成因。']
    (out/'实验报告.md').write_text('\n\n'.join(lines),'utf8'); save(out/'COMPLETE.json',{'best_epoch':best['epoch'],'epochs':len(history),'metrics':results})
def run(name):
    out=ROOT/name; out.mkdir(exist_ok=True)
    if (out/'COMPLETE.json').exists(): return
    manifest=json.loads((DATA/'manifest.json').read_text('utf8')); trainrows=[r for r in manifest['train'] if r['异常代码'] in CODES and r['二分类标签']=='1']; valrows=[r for r in manifest['val'] if r['异常代码'] in CODES and r['二分类标签']=='1']
    seed(42); m=Dual(); initial_head={k:v.clone() for k,v in m.head[2].state_dict().items()}
    if name in ['B0','B2']:
        state=torch.load(DATA/'a8.pt',map_location='cpu',weights_only=False)['model']; state.pop('head.2.weight'); state.pop('head.2.bias'); missing=m.load_state_dict(state,strict=False); assert set(missing.missing_keys)=={'head.2.weight','head.2.bias'} and not missing.unexpected_keys
    else:
        state=torch.load(DATA/'imagenet.pth',map_location='cpu',weights_only=True); state.pop('classifier.2.weight'); state.pop('classifier.2.bias'); m.net.load_state_dict(state,strict=True)
    assert all(torch.equal(v,initial_head[k]) for k,v in m.head[2].state_dict().items())
    freeze_info=None
    if name=='B2':
        audit=json.loads((DATA/'data_audit.json').read_text('utf8'))
        for path in [DATA/'a8.pt',ROOT.parent/'A/A8_高分辨率实验/模型/best.pt']:
            assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['a8.pt'],str(path)
        seen=set()
        for split,filename in [('train','训练集.csv'),('val','验证集.csv'),('test','测试集.csv')]:
            assert hashlib.sha256((ROOT.parents[1]/'数据划分'/filename).read_bytes()).hexdigest()==audit[split]['sha256']
            rs=manifest[split]; ids={r['钱币编号'] for r in rs}; assert len(ids)==len(rs) and not ids&seen; seen |= ids
            actual={c:sum(r['二分类标签']=='1' and r['异常代码']==c for r in rs) for c in CODES}; assert actual==audit[split]['six_counts']
            assert all((DATA/p).is_file() for r in rs for p in r['images'])
        before=sum(p.numel() for p in m.parameters() if p.requires_grad)
        # torchvision features: stem=0; stages=1,3,5,7; downsampling=2,4,6.
        assert len(m.net.features)==8
        for block in list(m.net.features)[:7]: block.requires_grad_(False)
        trainable=[n for n,p in m.named_parameters() if p.requires_grad]
        assert all(n.startswith(('net.features.7.','net.classifier.0.','head.')) for n in trainable)
        freeze_info={'before_trainable':before,'after_trainable':sum(p.numel() for p in m.parameters() if p.requires_grad),'frozen_parameters':sum(p.numel() for p in m.parameters() if not p.requires_grad),'frozen_modules':'net.features.0 through 6 inclusive (stem, stages1-3, all three downsampling layers)','trainable_modules':'net.features.7 (stage4), final LayerNorm net.classifier.0, head.0 and head.2','frozen_train_mode':'unchanged from B0; stochastic depth retained, ConvNeXt has no BatchNorm running statistics','trainable_names':trainable,'initial_frozen_sha256':frozen_hash(m),'initialization_sha256':audit['a8.pt'],'missing_keys':list(missing.missing_keys),'unexpected_keys':list(missing.unexpected_keys)}
        save(out/'freeze_verification.json',freeze_info)
    m.cuda(); opt=torch.optim.AdamW([{'params':[p for p in m.net.parameters() if p.requires_grad],'lr':1e-5},{'params':m.head.parameters(),'lr':1e-4}],weight_decay=.01)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=4,threshold=0,threshold_mode='abs',min_lr=[1e-7,1e-6])
    counts=np.bincount([CODES.index(r['异常代码']) for r in trainrows],minlength=6); weights=1/np.sqrt(counts); weights/=weights.mean(); w=torch.tensor(weights,dtype=torch.float32,device='cuda')
    cfg={'experiment':name,'initialization':'A8 binary backbone + hidden MLP; new six-class output' if name=='B0' else 'ImageNet backbone + seeded random MLP','classes':dict(zip(CODES,NAMES)),'class_counts':counts.tolist(),'class_weights':weights.tolist(),'seed':42,'resolution':512,'batch':4,'accumulation':2,'effective_batch':8,'backbone_lr':1e-5,'head_lr':1e-4,'max_epochs':60,'patience':10,'selection':'val Macro-F1; earliest epoch breaks ties','scheduler':'ReduceLROnPlateau factor .5 patience4','amp':'bfloat16','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'data_audit':json.loads((DATA/'data_audit.json').read_text('utf8')),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}; save(out/'config.json',cfg)
    cfg.update(execution_host='local Windows only',data_directory=str(DATA.resolve()),output_directory=str(out.resolve()))
    if name=='B2':
        cfg.update(initialization='A8 binary backbone + hidden MLP; new six-class output',freeze=freeze_info,weight_decay=.01,gradient_clip_norm=5)
        base=json.loads((ROOT/'B0/config.json').read_text('utf8'))
        for key in ['classes','class_counts','class_weights','seed','resolution','batch','accumulation','effective_batch','backbone_lr','head_lr','max_epochs','patience','selection','scheduler','amp','gpu','torch','data_audit']:
            assert cfg[key]==base[key],key
    save(out/'config.json',cfg)
    print(f'{name} START gpu={torch.cuda.get_device_name()} train={len(trainrows)} val={len(valrows)} batch=4 accumulation=2',flush=True)
    history=[]; best=-1.; stale=0; start=1
    if (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False); m.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); sched.load_state_dict(ck['scheduler']); history=ck['history']; best=ck['best']; stale=ck['stale']; start=ck['epoch']+1
    ds=Coins(trainrows,True); val=Coins(valrows)
    for epoch in range(start,61):
        if stale>=10: break
        seed(42+epoch); ds.epoch=epoch; dl=loader(ds,True); m.train(); t=time.time(); total=0.; denom=0.; yy=[]; pp=[]; it=iter(dl)
        for offset in range(0,len(dl),2):
            batches=[next(it) for _ in range(min(2,len(dl)-offset))]; group_den=sum(w[y.cuda()].sum().item() for _,y,_ in batches); opt.zero_grad(set_to_none=True)
            for x,y,_ in batches:
                y=y.cuda(non_blocking=True)
                with torch.autocast('cuda',dtype=torch.bfloat16): z=m(x.cuda(non_blocking=True))
                loss=nn.functional.cross_entropy(z.float(),y,weight=w,reduction='sum'); assert torch.isfinite(loss); (loss/group_den).backward(); total+=loss.item(); denom+=w[y].sum().item(); yy.extend(y.cpu().tolist()); pp.extend(z.detach().float().softmax(1).cpu().tolist())
            nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True); opt.step()
            if name=='B2' and offset==0:
                assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
                assert frozen_hash(m)==freeze_info['initial_frozen_sha256']
                freeze_info.update(verified_after_optimizer_epoch=epoch,frozen_gradients_none=True,frozen_weights_unchanged=True)
                save(out/'freeze_verification.json',freeze_info)
            if offset==0 or (offset+2)%80==0:
                print(f'{name} epoch={epoch} batches={min(offset+2,len(dl))}/{len(dl)} loss={total/denom:.6f} seconds={time.time()-t:.1f}',flush=True)
        vy,vp,vl=predict(m,val,w); f=metrics(vy,vp)['macro_f1']; improved=f>best
        if improved: best=f; stale=0; atomic({'model':m.state_dict(),'epoch':epoch,'val_macro_f1':f,'config':cfg},out/'best.pt')
        else: stale+=1
        sched.step(f); h={'epoch':epoch,'train_loss':total/denom,'val_loss':vl,'train_macro_f1':metrics(np.array(yy),np.array(pp))['macro_f1'],'val_macro_f1':f,'seconds':time.time()-t,'stale':stale,'backbone_lr':opt.param_groups[0]['lr']}; history.append(h)
        atomic({'model':m.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'epoch':epoch,'history':history,'best':best,'stale':stale},out/'last.pt'); save(out/'history.json',history); csvwrite(out/'history.csv',history); print(name,json.dumps(h),flush=True)
    if name=='B2':
        assert frozen_hash(m)==freeze_info['initial_frozen_sha256']
        freeze_info.update(final_frozen_sha256=frozen_hash(m),verified_after_all_epochs=True)
        save(out/'freeze_verification.json',freeze_info)
    finish(m,out,manifest,w,history,cfg)
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('experiment',choices=['B0','B1','B2']); args=p.parse_args(); run(args.experiment)
