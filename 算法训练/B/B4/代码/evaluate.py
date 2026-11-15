import json,csv,time
from pathlib import Path
import numpy as np
import torch
import b0_reference as b

@torch.inference_mode()
def collect(m,ds,w):
    m.eval();ys=[];ps=[];fs=[];total=den=0.
    def capture(module,args,out):fs.append(out.detach().float().cpu().numpy())
    hook=m.head[1].register_forward_hook(capture)
    for x,y,_ in b.loader(ds):
        with torch.autocast('cuda',dtype=torch.bfloat16):z=m(x.cuda(non_blocking=True))
        yy=y.cuda();total+=torch.nn.functional.cross_entropy(z.float(),yy,weight=w,reduction='sum').item();den+=w[yy].sum().item()
        ys.extend(y.tolist());ps.extend(z.float().softmax(1).cpu().tolist())
    hook.remove()
    return dict(y=np.array(ys),p=np.array(ps),features=np.concatenate(fs),loss=np.array(total/den))

def finish_all(m,out,manifest,w,history,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model'])
    results={};datasets={};arrays={}
    for split in ['train','val','test']:
        rows=[r for r in manifest[split] if r['异常代码'] in b.CODES and r['二分类标签']=='1'];ds=b.Coins(rows);datasets[split]=ds
        cache=out/f'{split}_final_inference.npz'
        if cache.exists():
            a=dict(np.load(cache));assert int(a['epoch'])==ck['epoch']
        else:
            a=collect(m,ds,w);np.savez_compressed(cache,**a,epoch=ck['epoch'])
        arrays[split]=a;result=b.metrics(a['y'],a['p']);result['loss']=float(a['loss']);results[split]=result;b.save(out/f'{split}_metrics.json',result)
        b.csvwrite(out/f'{split}_predictions.csv',[{**{k:v for k,v in r.items() if k!='images'},'true_class':b.CODES[a['y'][i]],'pred_class':b.CODES[a['p'][i].argmax()],**{f'p_{c}':float(a['p'][i,j]) for j,c in enumerate(b.CODES)}} for i,r in enumerate(rows)])
        b.plot_finish(out,ds,a['y'],a['p'],split)
        print('final inference',split,result['macro_f1'],flush=True)
    # Forward is exactly the B0 classifier; centers can be omitted in deployment.
    b.atomic(dict(model={k:v for k,v in ck['model'].items() if k!='centers'},epoch=ck['epoch'],config=cfg),out/'inference.pt')
    baseline=b.Dual().cuda();baseline.load_state_dict(torch.load(out.parent/'B0/best.pt',map_location='cpu',weights_only=False)['model'])
    baseline_arrays={}
    for split in ['train','val']:
        cache=out/f'b0_{split}_features.npz'
        if cache.exists():a=dict(np.load(cache))
        else:a=collect(baseline,datasets[split],w);np.savez_compressed(cache,**a)
        baseline_arrays[split]=a
    analysis={};fig,axs=plt.subplots(2,2,figsize=(12,10))
    for col,(name,aa) in enumerate([('B0',baseline_arrays),('B4',arrays)]):
        z=aa['train']['features'];z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12);y=aa['train']['y']
        means=np.stack([z[y==i].mean(0) for i in range(6)]);unit=means/np.linalg.norm(means,axis=1,keepdims=True)
        pca=PCA(n_components=2).fit(z)
        analysis[name]={'train_centroid_cosine':(unit@unit.T).tolist(),'pca_explained_variance_ratio':pca.explained_variance_ratio_.tolist()}
        for row,split in enumerate(['train','val']):
            a=aa[split];f=a['features'];f=f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-12);y=a['y'];xy=pca.transform(f)
            items={}
            for i,c in enumerate(b.CODES):
                v=f[y==i];dist=((v-unit[i])**2).sum(1);cos=v@unit.T;other=np.max(np.where(np.arange(6)==i,-np.inf,cos),axis=1)
                items[c]=dict(n=len(v),mean_squared_distance_to_train_centroid=float(dist.mean()),mean_cosine_margin=float((cos[:,i]-other).mean()),nearest_train_centroid_accuracy=float((cos.argmax(1)==i).mean()))
                axs[row,col].scatter(xy[y==i,0],xy[y==i,1],s=12,alpha=.45,label=c)
            analysis[name][split]=items;axs[row,col].set_title(f'{name} {split} (PCA fitted on own train)');axs[row,col].legend()
    fig.tight_layout();fig.savefig(out/'feature_PCA.png',dpi=150);plt.close(fig)
    centers=ck['model']['centers'].float().numpy();centers/=np.linalg.norm(centers,axis=1,keepdims=True)
    analysis['learned_center_cosine']=(centers@centers.T).tolist()
    for split in ['train','val']:
        a=arrays[split];f=a['features'];f/=np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-12)
        analysis[f'{split}_learned_center_distance']={c:float(((f[a['y']==i]-centers[i])**2).sum(1).mean()) for i,c in enumerate(b.CODES)}
    b.save(out/'feature_analysis.json',analysis)
    fig,axs=plt.subplots(1,3,figsize=(15,4))
    for ax,name,cm in zip(axs,['B0 empirical centers','B4 empirical centers','B4 learned centers'],[analysis['B0']['train_centroid_cosine'],analysis['B4']['train_centroid_cosine'],analysis['learned_center_cosine']]):
        im=ax.imshow(cm,vmin=-1,vmax=1,cmap='coolwarm');ax.set(xticks=range(6),yticks=range(6),xticklabels=b.CODES,yticklabels=b.CODES,title=name)
        for i in range(6):
            for j in range(6):ax.text(j,i,f'{cm[i][j]:.2f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=axs.tolist(),shrink=.7);fig.savefig(out/'center_cosine.png',dpi=150);plt.close(fig)
    ep=[h['epoch'] for h in history];fig,axs=plt.subplots(1,3,figsize=(16,4))
    for key in ['train_ce','val_loss','train_total_loss']:axs[0].plot(ep,[h[key] for h in history],label=key)
    axs[1].plot(ep,[h['train_center_loss'] for h in history],label='train normalized center loss')
    axs[1].plot(ep,[h['val_center_loss'] for h in history],label='val normalized center loss')
    for key in ['train_macro_f1','val_macro_f1']:axs[2].plot(ep,[h[key] for h in history],label=key)
    for ax in axs:ax.legend();ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=150);plt.close(fig)
    b.save(out/'COMPLETE.json',dict(best_epoch=ck['epoch'],epochs=len(history),metrics=results))
    from report import main
    main()
