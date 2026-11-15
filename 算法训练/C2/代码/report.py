import json,time
from pathlib import Path
import numpy as np
import torch
def complete_report(m,out,manifest,w,history,cfg):
    import train as t
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    C0=out.parent/'C/C0'
    read=lambda p:json.loads(p.read_text('utf8'))
    train=t.Coins([r for r in manifest['train'] if r['label']>=0])
    ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model'])
    y,p,loss=t.b.predict(m,train,w);c2train=t.enriched(y,p);t.save(out/'best_train_eval_metrics.json',c2train)
    t.wc(out/'best_train_eval_predictions.csv',[dict(coin_id=r['钱币编号'],true_grade=t.LABELS[int(y[i])],pred_grade=t.LABELS[int(p[i].argmax())]) for i,r in enumerate(train.rows)])
    # C0 checkpoint is used ONLY for a post-training comparable train evaluation.
    old=torch.load(C0/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(old['model'])
    y,p,loss=t.b.predict(m,train,w);c0train=t.enriched(y,p);t.save(out/'C0_best_train_eval_metrics.json',c0train)
    ep=[h['epoch'] for h in history];fig,axs=plt.subplots(1,3,figsize=(17,4))
    for ax,keys in zip(axs,[['train_loss','train_eval_loss','val_loss'],['train_accuracy','train_eval_accuracy','val_accuracy'],['train_macro_f1','train_eval_macro_f1','val_macro_f1']]):
        for key in keys:ax.plot(ep,[h[key] for h in history],label=key)
        ax.legend(fontsize=8);ax.set_xlabel('Epoch')
    fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
    c0v=read(C0/'val_metrics.json');c0t=read(C0/'test_metrics.json');v=read(out/'val_metrics.json');test=read(out/'test_metrics.json');flow=read(out/'test_pipeline_metrics.json')
    rows=[]
    for label in t.LABELS:
        rows.append(dict(grade=label,train_support=cfg['class_counts'][t.LABELS.index(label)],test_support=test['per_class'][label]['support'],C0_f1=c0t['per_class'][label]['f1'],C2_f1=test['per_class'][label]['f1'],delta=test['per_class'][label]['f1']-c0t['per_class'][label]['f1']))
    t.wc(out/'C0_C2_per_grade.csv',rows)
    pairs=[]
    for split in ['val','test']:
        a=np.array(read(C0/f'{split}_metrics.json')['confusion_matrix']);b=np.array(read(out/f'{split}_metrics.json')['confusion_matrix'])
        for i in range(t.K):
            for j in range(t.K):
                if i!=j and (a[i,j] or b[i,j]):pairs.append(dict(split=split,true_grade=t.LABELS[i],pred_grade=t.LABELS[j],adjacent=abs(i-j)==1,C0=int(a[i,j]),C2=int(b[i,j]),delta=int(b[i,j]-a[i,j])))
    t.wc(out/'C0_C2_confusion_pairs.csv',pairs)
    pct=lambda x:f'{x*100:.2f}%'
    besth=next(h for h in history if h['epoch']==ck['epoch']);oldh=next(h for h in read(C0/'history.json') if h['epoch']==old['epoch'])
    gap0=c0train['macro_f1']-c0v['macro_f1'];gap2=c2train['macro_f1']-v['macro_f1']
    timing=dict(best_epoch=ck['epoch'],epochs=len(history),early_stopped=history[-1]['stale']>=10,training_and_epoch_eval_seconds=sum(h['seconds'] for h in history),note='C2 includes extra epoch-end train eval; C0 historical time does not. Queue wait excluded.')
    t.save(out/'timing.json',timing)
    lines=['# C2 实验报告','唯一模型训练差异：部分冻结。原始 A8 初始化、C0 数据划分/20类/增广/优化器/学习率调度/选模条件复用。冻结层保留 C0 train 模式随机深度，不额外修改正则化。',f'参数：总计 {cfg["freeze"]["total_parameters"]:,}；可训练 {cfg["freeze"]["trainable_parameters"]:,}（{pct(cfg["freeze"]["trainable_ratio"])}）。冻结参数梯度、首次更新和最终不变性均经过断言。',f'最佳轮次 {ck["epoch"]}；实际 {len(history)} 轮；早停 {timing["early_stopped"]}；训练及逐轮评估 {timing["training_and_epoch_eval_seconds"]/60:.1f} 分钟。等待时间不计入。','|集合|Accuracy|20类Macro-F1|数值MAE|四档Accuracy|四档Macro-F1|','|---|---:|---:|---:|---:|---:|']
    for name,r in [('train(best,eval)',c2train),('val',v),('test',test)]:lines.append(f'|{name}|{pct(r["accuracy"])}|{pct(r["macro_f1"])}|{r["numeric_mae"]:.3f}|{pct(r["four_group"]["accuracy"])}|{pct(r["four_group"]["macro_f1"])}|')
    lines+=['Macro-F1始终固定20类，零支持类记0。四档为细类概率求和。train_eval 指标使用轮末固定模型、无增强、eval，与验证统计口径一致；原 train_macro_f1 来自增强及轮内变化的模型。训练和验证支持分布不同，因此泛化差距仍受稀有/无支持类影响。',f'A8冻结分流：普通币端到端Accuracy={pct(flow["ordinary_end_to_end_accuracy"])}；普通币Macro-F1（拒绝计漏判）={pct(flow["ordinary_macro_f1_with_A_rejections"])}；计入异常误入的等级Macro-F1={pct(flow["all_flow_grade_macro_f1"])}。完整覆盖和拒绝计数见 test_pipeline_metrics.json。','|等级|训练支持|测试支持|C0 F1|C2 F1|变化百分点|','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f'|{r["grade"]}|{r["train_support"]}|{r["test_support"]}|{pct(r["C0_f1"])}|{pct(r["C2_f1"])}|{r["delta"]*100:+.2f}|')
    (out/'实验报告.md').write_text('\n\n'.join(lines),'utf8')
    dv=v['macro_f1']-c0v['macro_f1'];dt=test['macro_f1']-c0t['macro_f1']
    small=[r for r in rows if r['train_support']<100 and r['test_support']>0]
    small_delta=float(np.mean([r['delta'] for r in small]))
    comparison=['# C0_C2 对比报告',f'同口径最佳模型训练/验证Macro-F1：C0 {pct(c0train["macro_f1"])} / {pct(c0v["macro_f1"])}（差 {gap0*100:.2f} 点）；C2 {pct(c2train["macro_f1"])} / {pct(v["macro_f1"])}（差 {gap2*100:.2f} 点）。'+('冻结缩小了训练—验证差距。' if gap2<gap0 else '冻结未缩小训练—验证差距。')+'差距缩小本身不能证明泛化改善，需结合验证和测试。',f'保留C0原报告口径：最佳轮轮内增强训练Macro-F1={pct(oldh["train_macro_f1"])}；C2同口径={pct(besth["train_macro_f1"])}。不得将其与新增eval训练指标混算。',f'验证Macro-F1：{pct(c0v["macro_f1"])} → {pct(v["macro_f1"])}（{dv*100:+.2f}点）；测试Macro-F1：{pct(c0t["macro_f1"])} → {pct(test["macro_f1"])}（{dt*100:+.2f}点）。单seed固定划分的观察，不作显著性结论。',f'少数类定义为训练支持<100且测试有支持；这些类别平均F1变化 {small_delta*100:+.2f} 点。逐类数据及AU53/AU55/AU58、MS61—MS66见实验报告和C0_C2_per_grade.csv；无测试支持不用于少数类均值。']
    for name,r0,r2 in [('验证',c0v,v),('测试',c0t,test)]:
        comparison.append(f'{name}相邻等级错误：{r0["adjacent_label_errors"]} → {r2["adjacent_label_errors"]}；占全部样本 {pct(r0["adjacent_label_errors"]/r0["n"])} → {pct(r2["adjacent_label_errors"]/r2["n"])}；总错误 {r0["errors"]} → {r2["errors"]}；数值MAE {r0["numeric_mae"]:.3f} → {r2["numeric_mae"]:.3f}。相邻定义为排序标签索引差1，详细方向见混淆对比CSV。')
    if dv>0 and dt>0 and gap2<gap0:
        advice='下一步优先继续优化泛化：冻结在验证和测试均有收益，可先验证收益的稳定性，再决定是否引入有序目标。'
    elif test['adjacent_label_errors']/max(test['errors'],1)>.5:
        advice='下一步更值得考虑C3有序分类：冻结未同时满足泛化提升和差距缩小，且过半错误仍在相邻等级。C3需预先固定协议，不能根据本轮测试反复调参。'
    else:advice='下一步优先继续诊断泛化与稀有类支持；当前错误尚不足以支持直接优先有序分类。'
    comparison += [advice,'本轮只运行C2；未启动C3/C4。C0最佳权重仅在C2训练完成后用于同口径训练集评估，从未用于C2初始化。测试只在验证选模冻结后评价。']
    (out/'C0_C2对比报告.md').write_text('\n\n'.join(comparison),'utf8')
    t.save(out/'COMPLETE.json',dict(**timing,completed_at=time.time(),test=test,comparison_report='C0_C2对比报告.md'))
