import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,csv,time,re,hashlib,importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
from torchvision import transforms as T
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
R=Path(__file__).resolve().parents[1]; DATA=R.parent/'B/B0/数据缓存'
spec=importlib.util.spec_from_file_location('b_reference',Path(__file__).resolve().parent/'b_reference.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
save=b.save;wc=b.csvwrite
LABELS=list(json.loads((R/'label_mapping.json').read_text('utf8')));K=len(LABELS);NUM=np.array([int(re.search(r'\d+',g)[0]) for g in LABELS]);GROUP=np.array([0 if n<=35 else 1 if n<=45 else 2 if n<=58 else 3 for n in NUM])
class Coins(b.Coins):
    def __getitem__(self,i):
        if self.augment:torch.manual_seed(42+self.epoch*100000+i)
        r=self.rows[i];ims=[]
        for p in r['images']:
            with Image.open(p) as im:ims.append(self.transform(im.convert('RGB').resize((512,512),Image.Resampling.BILINEAR)))
        return torch.stack(ims),r['label'],i
class Dual(b.Dual):
    def __init__(self):
        super().__init__();self.head[2]=nn.Linear(256,K)
def metrics(y,p,labels=LABELS):
    pred=p.argmax(1);a,r,f,n=precision_recall_fscore_support(y,pred,labels=range(len(labels)),zero_division=0)
    return dict(n=len(y),accuracy=float((y==pred).mean()),macro_f1=float(f.mean()),per_class={g:dict(precision=float(a[i]),recall=float(r[i]),f1=float(f[i]),support=int(n[i])) for i,g in enumerate(labels)},confusion_labels=labels,confusion_matrix=confusion_matrix(y,pred,labels=range(len(labels))).tolist())
def enriched(y,p):
    result=metrics(y,p);pred=p.argmax(1);err=y!=pred
    result.update(numeric_mae=float(np.abs(NUM[y]-NUM[pred]).mean()),errors=int(err.sum()),adjacent_label_errors=int((np.abs(y[err]-pred[err])==1).sum()),within_3_numeric_errors=int((np.abs(NUM[y[err]]-NUM[pred[err]])<=3).sum()))
    gp=np.stack([p[:,GROUP==g].sum(1) for g in range(4)],1);result['four_group']=metrics(GROUP[y],gp,['VF及以下','XF','AU','MS']);return result
def finish(m,out,manifest,w,history,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);results={};flows={}
    # Freeze validation selection before any test inference.
    save(out/'frozen_selection.json',dict(best_epoch=ck['epoch'],val_macro_f1=ck['val_macro_f1'],selection=cfg['selection'],config_sha256=hashlib.sha256((out/'config.json').read_bytes()).hexdigest()))
    for split in ['val','test']:
        ds=Coins(manifest[split]);cache=out/f'{split}_final_inference.npz'
        if cache.exists():
            cached=np.load(cache);assert int(cached['epoch'])==ck['epoch'];y,p,loss=cached['y'],cached['p'],float(cached['loss'])
        else:
            y,p,loss=b.predict(m,ds,w);np.savez_compressed(cache,y=y,p=p,loss=loss,epoch=ck['epoch'])
        mask=y>=0;result=enriched(y[mask],p[mask]);result['loss']=loss;results[split]=result;save(out/f'{split}_metrics.json',result)
        with (DATA/f'a8_{split}.csv').open(encoding='utf-8-sig') as f:binary={r['钱币编号']:r for r in csv.DictReader(f)}
        assert set(binary)=={r['钱币编号'] for r in ds.rows}
        thresholds={float(r['threshold']) for r in binary.values()};assert thresholds=={float(json.loads((DATA/'a8_threshold.json').read_text('utf8'))['threshold'])}
        route=np.array([int(binary[r['钱币编号']]['pred_label'])==0 for r in ds.rows]);normal=np.array([r['二分类标签']=='0' for r in ds.rows]);special=np.array([r['category']=='SP/PR' for r in ds.rows]);pred=p.argmax(1)
        result.update(normal_total=int(normal.sum()),ordinary_covered=int(mask.sum()),sp_pr=int(special.sum()),other_normal=int((normal&~mask&~special).sum()),all_normal_direct_correct_covered_rate=float((mask&(pred==y)).sum()/normal.sum()),coverage_note='Fine-grade metrics cover ordinary numeric grades; all real normals enter C, with unsupported normals retained in predictions and coverage counts.')
        save(out/f'{split}_metrics.json',result)
        covered_correct=mask&route&(pred==y);flowpred=np.where(route,pred,K)
        _,_,ff,_=precision_recall_fscore_support(y[mask],flowpred[mask],labels=range(K),zero_division=0)
        # Precision over all routed coins also counts abnormal and unsupported normals as false positives.
        ally=np.where(mask,y,K);aa,bb,cc,nnn=precision_recall_fscore_support(ally,flowpred,labels=range(K),zero_division=0)
        flow=dict(threshold=next(iter(thresholds)),source='frozen A8 single-model predictions',all_coins=len(y),normal_total=int(normal.sum()),ordinary_covered=int(mask.sum()),normal_rejected_by_A=int((normal&~route).sum()),ordinary_rejected_by_A=int((mask&~route).sum()),ordinary_end_to_end_accuracy=float(covered_correct.sum()/mask.sum()),normal_all_denominator_correct_covered_rate=float(covered_correct.sum()/normal.sum()),ordinary_macro_f1_with_A_rejections=float(ff.mean()),all_flow_grade_macro_f1=float(cc.mean()),all_flow_per_grade={g:dict(precision=float(aa[i]),recall=float(bb[i]),f1=float(cc[i]),support=int(nnn[i])) for i,g in enumerate(LABELS)},abnormal_total=int((~normal).sum()),abnormal_entering_C=int((~normal&route).sum()),sp_pr_total=int(special.sum()),sp_pr_entering_C=int((special&route).sum()),other_normal_total=int((normal&~mask&~special).sum()),coverage_note='SP/PR and other unsupported normals have no correct C output; all-normal rate includes them in denominator. Abnormal entrants count as grade false positives in all-flow precision.')
        flows[split]=flow;save(out/f'{split}_pipeline_metrics.json',flow)
        records=[]
        for i,r in enumerate(ds.rows):
            records.append({**{k:v for k,v in r.items() if k!='images'},'pred_grade':LABELS[pred[i]],'A_anomaly_probability':binary[r['钱币编号']]['coin_score'],'A_pred_label':int(not route[i]),'pipeline_pred':LABELS[pred[i]] if route[i] else 'abnormal','C_covered_correct':bool(mask[i] and pred[i]==y[i]),'pipeline_covered_correct':bool(covered_correct[i]),**{f'p_{g}':float(p[i,j]) for j,g in enumerate(LABELS)},**{f'p_group_{g}':float(p[i,GROUP==j].sum()) for j,g in enumerate(['VF及以下','XF','AU','MS'])}})
        wc(out/f'{split}_all_pipeline_predictions.csv',records);wc(out/f'{split}_predictions.csv',[r for r in records if r['二分类标签']=='0'])
        cm=np.array(result['confusion_matrix']);wc(out/f'{split}_confusion.csv',[dict(true_grade=g,**dict(zip(LABELS,cm[i].tolist()))) for i,g in enumerate(LABELS)])
        fig,ax=plt.subplots(figsize=(12,10));ax.imshow(cm,cmap='Blues');ax.set(xticks=range(K),yticks=range(K),xticklabels=LABELS,yticklabels=LABELS,xlabel='Predicted',ylabel='True');plt.setp(ax.get_xticklabels(),rotation=90)
        for i in range(K):
            for j in range(K):
                if cm[i,j]:ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7)
        fig.tight_layout();fig.savefig(out/f'{split}_confusion.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for key in ['train_loss','val_loss']:axs[0].plot([h['epoch'] for h in history],[h[key] for h in history],label=key)
    for key in ['train_macro_f1','val_macro_f1']:axs[1].plot([h['epoch'] for h in history],[h[key] for h in history],label=key)
    for ax in axs:ax.legend();ax.set_xlabel('Epoch')
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=150);plt.close(fig)
    cm=np.array(results['val']['confusion_matrix']);np.fill_diagonal(cm,0);pairs=sorted([(int(cm[i,j]),LABELS[i],LABELS[j]) for i in range(K) for j in range(K) if cm[i,j]],reverse=True)[:8]
    lines=[f'# {out.name} 实验报告',f'目的：检验阶段一初始化对具体基础品相的帮助。初始化：{cfg["initialization"]}。部分冻结微调；迁移和冻结范围见 initialization.json 与 freeze_verification.json。',f'数据：{cfg["ordinary_counts"]}；K={K}。保留稀有类；固定全类别 Macro-F1（零支持类别计0）。不预测plus；SP/PR单列。',f'最佳轮次：{ck["epoch"]}；实际训练 {len(history)} 轮。验证集选模，测试仅在冻结后评价。', '|集合|Accuracy|Macro-F1|数值MAE|四档Accuracy|','|---|---:|---:|---:|---:|']
    for s,v in results.items():lines.append(f'|{s}|{v["accuracy"]:.4f}|{v["macro_f1"]:.4f}|{v["numeric_mae"]:.3f}|{v["four_group"]["accuracy"]:.4f}|')
    lines += [f'验证主要混淆（数量、真实、预测）：{pairs}',f'测试流程及覆盖分母：{json.dumps(flows["test"],ensure_ascii=False)}','A3 四档参考验证77.34%、测试78.38%；A3按异常AUROC选模，与本轮具体等级任务不同。','每类P/R/F1及支持量见 val_metrics.json/test_metrics.json；顺序混淆矩阵、逐币全类别概率另存。']
    (out/'实验报告.md').write_text('\n\n'.join(lines),'utf8');save(out/'EVALUATED.json',dict(best_epoch=ck['epoch'],epochs=len(history),metrics=results))

def freeze(m):
    for p in m.net.features[:7].parameters(): p.requires_grad_(False)
    total=sum(p.numel() for p in m.parameters());trainable=sum(p.numel() for p in m.parameters() if p.requires_grad)
    return dict(total_parameters=total,trainable_parameters=trainable,trainable_ratio=trainable/total,frozen_modules='net.features.0..6: stem, stages 1-3 and all downsampling',trainable_modules='net.features.7 stage4, net.classifier.0 final norm, head.0 fusion and head.2 classification',frozen_train_mode='C0 unchanged: stochastic depth remains active in train mode; no BatchNorm',trainable_names=[n for n,p in m.named_parameters() if p.requires_grad])
def verify_updates(m,frozen,trainable,path):
    assert all(p.grad is None for n,p in m.named_parameters() if n in frozen)
    unchanged=all(torch.equal(p.detach().cpu(),frozen[n]) for n,p in m.named_parameters() if n in frozen)
    changed=[n for n,p in m.named_parameters() if n in trainable and not torch.equal(p.detach().cpu(),trainable[n])]
    assert unchanged and len(changed)==len(trainable), (unchanged, set(trainable)-set(changed))
    save(path,dict(frozen_unchanged=unchanged,frozen_gradients_none=True,all_trainable_tensors_updated=True,changed=changed))

def run(name):
    assert name==R.name, 'Run each experiment from its own directory'
    out=R
    if (out/'COMPLETE.json').exists():return
    manifest=json.loads((R/'manifest.json').read_text('utf8'));train=[r for r in manifest['train'] if r['label']>=0];val=[r for r in manifest['val'] if r['label']>=0]
    b.seed(42);m=Dual();initial={k:v.clone() for k,v in m.head[2].state_dict().items()}
    source=DATA/('a8.pt' if name=='C2' else 'imagenet.pth')
    if name=='C2':
        state=torch.load(source,map_location='cpu',weights_only=False)['model'];state.pop('head.2.weight');state.pop('head.2.bias');missing=m.load_state_dict(state,strict=False);assert set(missing.missing_keys)=={'head.2.weight','head.2.bias'} and not missing.unexpected_keys
    else:
        state=torch.load(source,map_location='cpu',weights_only=True);state.pop('classifier.2.weight');state.pop('classifier.2.bias');m.net.load_state_dict(state,strict=True)
    assert all(torch.equal(v,initial[k]) for k,v in m.head[2].state_dict().items())
    freeze_info=freeze(m)
    save(out/'initialization.json',dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),transferred='net + head.0' if name=='C2' else 'net',new='head.2' if name=='C2' else 'head.0 + head.2',output_reinitialized=True,all_parameters_trainable=all(p.requires_grad for p in m.parameters())))
    freeze_info=freeze(m);save(out/'freeze_verification.json',freeze_info)
    counts=np.bincount([r['label'] for r in train],minlength=K);weights=np.zeros(K);supported=counts>0;weights[supported]=1/np.sqrt(counts[supported]);weights[supported]/=weights[supported].mean()
    # No-train labels remain in evaluation and softmax. Unit evaluation weight avoids zero-loss masking.
    weights[~supported]=1.;m.cuda();w=torch.tensor(weights,dtype=torch.float32,device='cuda')
    cfg=dict(experiment=name,initialization='A8 single model backbone + fusion hidden' if name=='C2' else 'ImageNet backbone + random fusion hidden',classes=LABELS,class_counts=counts.tolist(),class_weights=weights.tolist(),unsupported_train=[LABELS[i] for i in range(K) if not supported[i]],ordinary_counts={s:sum(r['label']>=0 for r in rs) for s,rs in manifest.items()},seed=42,resolution=512,batch=4,accumulation=2,bf16=True,rotation=10,backbone_lr=1e-5,head_lr=1e-4,weight_decay=.01,max_epochs=60,patience=10,selection='validation Macro-F1 over fixed union K labels, zero_division=0; earliest epoch wins ties',scheduler='ReduceLROnPlateau max, factor .5 patience4 threshold0 abs min_lr1e-7/1e-6',execution='local GPU only',gpu=torch.cuda.get_device_name(),torch=torch.__version__,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());cfg.update(freeze=freeze_info,train_metric_protocol='train_macro_f1: original online augmented; train_eval_*: epoch-end eval without augmentation, same fixed 20 labels as validation');save(out/'config.json',cfg)
    for f in ['label_mapping.json','grade_counts.csv','data_audit.json']:(out/f).write_bytes((R/f).read_bytes())
    # Preflight happens only inside the GPU mutex, before formal training.
    ds=Coins(train,True);x,y,_=next(iter(torch.utils.data.DataLoader(ds,batch_size=4,num_workers=0)));assert x.shape==(4,2,3,512,512) and int(y.min())>=0 and int(y.max())<K
    m.train()
    with torch.autocast('cuda',dtype=torch.bfloat16):z=m(x.cuda())
    loss=nn.functional.cross_entropy(z.float(),y.cuda(),weight=w);assert torch.isfinite(loss);loss.backward();assert all((p.grad is not None and torch.isfinite(p.grad).all()) if p.requires_grad else p.grad is None for p in m.parameters());m.zero_grad(set_to_none=True)
    save(out/'preflight.json',dict(input_shape=list(x.shape),loss=loss.item(),finite_gradients=True,frozen_gradients_none=True,trainable_gradients_present=True));del x,y,z,loss
    opt=torch.optim.AdamW([{'params':[p for p in m.net.parameters() if p.requires_grad],'lr':1e-5},{'params':m.head.parameters(),'lr':1e-4}],weight_decay=.01);sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=4,threshold=0,threshold_mode='abs',min_lr=[1e-7,1e-6])
    initial_frozen={n:p.detach().cpu().clone() for n,p in m.named_parameters() if not p.requires_grad}
    initial_trainable={n:p.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad}
    history=[];best=-1.;stale=0;start=1
    if (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);sched.load_state_dict(ck['scheduler']);history=ck['history'];best=ck['best'];stale=ck['stale'];start=ck['epoch']+1
    for epoch in range(start,61):
        if stale>=10:break
        b.seed(42+epoch);ds.epoch=epoch;dl=b.loader(ds,True);m.train();t=time.time();total=0.;denom=0.;yy=[];pp=[];it=iter(dl)
        for offset in range(0,len(dl),2):
            batches=[next(it) for _ in range(min(2,len(dl)-offset))];group_den=sum(w[y.cuda()].sum().item() for _,y,_ in batches);opt.zero_grad(set_to_none=True)
            for x,y,_ in batches:
                y=y.cuda(non_blocking=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):z=m(x.cuda(non_blocking=True))
                loss=nn.functional.cross_entropy(z.float(),y,weight=w,reduction='sum');assert torch.isfinite(loss);(loss/group_den).backward();total+=loss.item();denom+=w[y].sum().item();yy.extend(y.cpu().tolist());pp.extend(z.detach().float().softmax(1).cpu().tolist())
            nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()
            if epoch==1 and offset==0:
                verify_updates(m,initial_frozen,initial_trainable,out/'first_update_verification.json')
            if offset%80==0:print(f'{name} epoch={epoch} batches={offset+len(batches)}/{len(dl)} seconds={time.time()-t:.1f}',flush=True)
        vy,vp,vl=b.predict(m,Coins(val),w);f=metrics(vy,vp)['macro_f1']
        if f>best:best=f;stale=0;b.atomic(dict(model=m.state_dict(),epoch=epoch,val_macro_f1=f,config=cfg),out/'best.pt')
        else:stale+=1
        ty,tp,tl=b.predict(m,Coins(train),w);tm=metrics(ty,tp);online=metrics(np.array(yy),np.array(pp));vm=metrics(vy,vp)
        sched.step(f);h=dict(epoch=epoch,train_loss=total/denom,val_loss=vl,train_macro_f1=metrics(np.array(yy),np.array(pp))['macro_f1'],val_macro_f1=f,seconds=time.time()-t,stale=stale,backbone_lr=opt.param_groups[0]['lr']);h.update(train_accuracy=online['accuracy'],val_accuracy=vm['accuracy'],train_eval_accuracy=tm['accuracy'],train_eval_macro_f1=tm['macro_f1'],train_eval_loss=tl);history.append(h)
        b.atomic(dict(model=m.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),epoch=epoch,history=history,best=best,stale=stale),out/'last.pt');save(out/'history.json',history);wc(out/'history.csv',history);print(json.dumps(h),flush=True)
        if len(history)==2:save(out/'timing_estimate.json',dict(mean_epoch_seconds=float(np.mean([h['seconds'] for h in history])),remaining_max_seconds=float(np.mean([h['seconds'] for h in history])*58),note='Estimate from first two actual epochs; early stopping may finish sooner.'))
    del opt,sched
    import gc
    gc.collect();torch.cuda.empty_cache();verify_updates(m,initial_frozen,initial_trainable,out/'final_update_verification.json');finish(m,out,manifest,w,history,cfg)
    from report import complete_report
    complete_report(m,out,manifest,w,history,cfg)
if __name__=='__main__':
    name=sys.argv[1] if len(sys.argv)>1 else R.name
    assert name==R.name;run(name)
