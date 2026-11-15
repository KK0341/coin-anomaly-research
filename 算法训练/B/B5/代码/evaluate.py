"""Validation diagnostics first; one optional, cached descriptive test pass."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import b0_reference as base
from model import Dual
from train import predict, loader
from prepare import digest

def confusion(out, y, p, name):
    cm = np.array(base.metrics(y,p)['confusion_matrix'])
    for normalized in [False, True]:
        a = cm / np.maximum(cm.sum(1,keepdims=True),1) if normalized else cm
        suffix = 'normalized' if normalized else 'counts'
        np.savetxt(out/f'{name}_confusion_{suffix}.csv', a, delimiter=',', header=','.join(base.CODES))
        fig,ax = plt.subplots(figsize=(7,6)); ax.imshow(a,cmap='Blues')
        for i in range(6):
            for j in range(6): ax.text(j,i,f'{a[i,j]:.2f}' if normalized else str(a[i,j]),ha='center')
        ax.set(xticks=range(6),yticks=range(6),xticklabels=base.CODES,yticklabels=base.CODES,xlabel='Predicted',ylabel='True')
        fig.tight_layout(); fig.savefig(out/f'{name}_confusion_{suffix}.png',dpi=150); plt.close(fig)

@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--final-test',action='store_true')
    a = p.parse_args(); a.batch=4; a.workers=4
    assert (a.out/'TRAINING_COMPLETE.json').exists()
    assert torch.cuda.device_count()==1 and '5090' not in torch.cuda.get_device_name()
    base.DATA=a.data; manifest=json.loads((a.data/'manifest.json').read_text('utf8'))
    ck=torch.load(a.out/'best.pt',map_location='cpu',weights_only=False); cfg=ck['config']
    dtype=torch.bfloat16 if cfg['amp']=='torch.bfloat16' else torch.float16
    m=Dual().cuda(); m.load_state_dict(ck['model']); m.eval()
    w=torch.tensor(cfg['class_weights'],device='cuda'); val=base.Coins(manifest['val'])
    results={}
    for mode in ['full','global','uniform']:
        y,prob,loss=predict(m,val,w,a,dtype,mode)
        results[mode]=dict(base.metrics(y,prob),loss=loss)
        confusion(a.out,y,prob,'val_'+mode)
        base.csvwrite(a.out/f'val_{mode}_predictions.csv', [dict(coin_id=r['钱币编号'],true=base.CODES[int(y[i])],pred=base.CODES[int(prob[i].argmax())],**{f'p_{c}':float(prob[i,j]) for j,c in enumerate(base.CODES)}) for i,r in enumerate(val.rows)])
    base.save(a.out/'validation_diagnostics.json',results)
    # Representatives are chosen from validation only: first correct and first error per class.
    y,prob,_=predict(m,val,w,a,dtype)
    folder=a.out/'attention'; folder.mkdir(exist_ok=True)
    for k in range(6):
        for correct in [True,False]:
            candidates=np.flatnonzero((y==k)&((prob.argmax(1)==y)==correct))
            if not len(candidates): continue
            i=int(candidates[0]); x,_,_=val[i]
            with torch.autocast('cuda',dtype=dtype): _,_,att=m(x[None].cuda(),details=True)
            maps=att[0].cpu().numpy(); np.savez_compressed(folder/f'{val.rows[i]["钱币编号"]}.npz',attention=maps)
            fig,axs=plt.subplots(6,2,figsize=(7,17))
            for query in range(6):
                vmax=float(maps[query].max())
                for side in range(2):
                    ax=axs[query,side]
                    with Image.open(a.data/val.rows[i]['images'][side]) as im: ax.imshow(im)
                    ax.imshow(maps[query,side],extent=(0,512,512,0),cmap='jet',alpha=.45,vmin=0,vmax=vmax)
                    ax.set_title(f'query {base.CODES[query]} side {side}; mass={maps[query,side].sum():.3f}'); ax.axis('off')
            fig.suptitle(f'{val.rows[i]["钱币编号"]} true={base.CODES[k]} pred={base.CODES[prob[i].argmax()]}')
            fig.tight_layout(); fig.savefig(folder/f'{val.rows[i]["钱币编号"]}.png',dpi=110); plt.close(fig)
    history=json.loads((a.out/'history.json').read_text('utf8'))
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for key in ['train_loss','val_loss']: axs[0].plot([h['epoch'] for h in history],[h[key] for h in history],label=key)
    for key in ['train_macro_f1','val_macro_f1']: axs[1].plot([h['epoch'] for h in history],[h[key] for h in history],label=key)
    for ax in axs: ax.legend(); ax.set_xlabel('Epoch')
    fig.tight_layout(); fig.savefig(a.out/'training_curves.png',dpi=150); plt.close(fig)
    # Identical GPU, precision, inputs, batch and warmup; GPU-only forward timing.
    b0=base.Dual().cuda(); b0.load_state_dict(torch.load(a.data/'b0.pt',map_location='cpu',weights_only=False)['model']); b0.eval()
    x=next(iter(loader(val,4,0)))[0].cuda(); timings={}
    for name,model in [('B0',b0),('B5',m)]:
        with torch.autocast('cuda',dtype=dtype):
            for _ in range(10): model(x)
            torch.cuda.synchronize(); start=time.perf_counter()
            for _ in range(50): model(x)
            torch.cuda.synchronize()
        timings[name]=(time.perf_counter()-start)*1000/(50*len(x))
    base.save(a.out/'inference_cost.json',dict(gpu=torch.cuda.get_device_name(),amp=str(dtype),batch=4,warmup=10,iterations=50,forward_ms_per_coin=timings))
    if a.final_test:
        cache=a.out/'test_final_inference.npz'; claim=a.out/'TEST_STARTED.json'
        if not cache.exists():
            # If interrupted after the claim, require inspection instead of silently retesting.
            with claim.open('x',encoding='utf8') as stream: json.dump({'best_sha256':digest(a.out/'best.pt')},stream)
            ds=base.Coins(manifest['test']); ty,tp,tl=predict(m,ds,w,a,dtype)
            np.savez_compressed(cache,y=ty,p=tp,loss=tl,best_sha256=digest(a.out/'best.pt'))
        cached=np.load(cache); assert str(cached['best_sha256'])==digest(a.out/'best.pt')
        ty,tp=cached['y'],cached['p']; base.save(a.out/'test_metrics.json',base.metrics(ty,tp)); confusion(a.out,ty,tp,'test')
        base.csvwrite(a.out/'test_predictions.csv',[dict(coin_id=r['钱币编号'],true=base.CODES[int(ty[i])],pred=base.CODES[int(tp[i].argmax())],**{f'p_{c}':float(tp[i,j]) for j,c in enumerate(base.CODES)}) for i,r in enumerate(manifest['test'])])
    baseline=json.loads((a.data/'b0_val_metrics.json').read_text('utf8'))
    def triplet(met):
        cm=np.array(met['confusion_matrix']); indices=[0,2,3]
        return int(sum(cm[i,j] for i in indices for j in indices if i!=j))
    comparison=dict(b0_val_macro_f1=baseline['macro_f1'],b5_val_macro_f1=results['full']['macro_f1'],
                    delta=results['full']['macro_f1']-baseline['macro_f1'],
                    b0_92_98_95_errors=triplet(baseline),b5_92_98_95_errors=triplet(results['full']))
    base.save(a.out/'b0_b5_comparison.json',comparison)
    text=['# B5 实验报告',f'最佳轮次：{ck["epoch"]}；代码版本：{cfg["git_commit"]}。',
          '共享ConvNeXt-Tiny，双面512输入；六个768维查询在双面共512个空间位置上softmax；类别证据线性打分与B0全局logit相加。新增9,222参数，无局部分支、无类别中心损失。',
          f'训练GPU：{cfg["gpu"]}；精度：{cfg["amp"]}；总轮次耗时：{sum(h["seconds"] for h in history):.1f}秒；峰值已分配显存：{max(h["peak_allocated_gib"] for h in history):.3f}GiB。',
          f'验证对比：{json.dumps(comparison,ensure_ascii=False)}',
          f'B5全局分支F1={results["global"]["macro_f1"]:.6f}；均匀聚合F1={results["uniform"]["macro_f1"]:.6f}；完整B5 F1={results["full"]["macro_f1"]:.6f}。',
          '关闭辅助分支及均匀聚合是同一已训练模型的推理消融，不等于独立重训的因果实验。注意力仅是可视化证据，不证明缺陷定位正确。',
          'B0使用本机RTX5060 Laptop和BF16。B5硬件及可能的混合精度差异见上文，因此不是完全严格单变量实验。速度比较在同一云端GPU和精度下完成，仅计GPU前向，不含数据加载。',
          '测试集已在既往实验使用；如有test_metrics.json，它仅为冻结模型后的一次描述性评价，不用于调参。',
          '下一轮最有价值的改进：在统一硬件、精度及多个预先固定随机种子下复验类别查询收益，先确认增益是否稳定。未启动B6。',
          '当前为云端生成报告；本机回传与SHA256验收状态应另行记录，不据此声称回传成功。']
    (a.out/'实验报告.md').write_text('\n\n'.join(text),'utf8')
    files={str(f.relative_to(a.out)):digest(f) for f in a.out.rglob('*') if f.is_file() and f.name not in ['last.pt','training.lock','result_sha256.json','train.log']}
    base.save(a.out/'result_sha256.json',files)

if __name__=='__main__': main()
