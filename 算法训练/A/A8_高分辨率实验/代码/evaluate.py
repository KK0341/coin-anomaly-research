import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

ROOT=Path(__file__).resolve().parents[1]; R=ROOT.parent; OUT=ROOT/'结果'
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def read(p): return pd.read_csv(p,dtype=str).fillna('')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def aligned(p,d,col='coin_score'):
    f=read(p); assert f['钱币编号'].is_unique and set(f['钱币编号'])==set(d['钱币编号'])
    f=f.set_index('钱币编号').loc[d['钱币编号']]
    if 'true_label' in f: assert np.array_equal(f.true_label.astype(int),d['二分类标签'].astype(int))
    for k in ['正面路径','背面路径']:
        if k in f: assert f[k].tolist()==d[k].tolist()
    return f[col].astype(float).to_numpy()
def threshold(y,s,point):
    if point=='0.5': return .5
    if point=='recall154': return float(np.sort(s[y==1])[::-1][153])
    p,r,t=precision_recall_curve(y,s); f=2*p[:-1]*r[:-1]/(p[:-1]+r[:-1]+1e-12)
    return float(t[np.flatnonzero(f==f.max())[-1]])
def measure(y,s,t):
    pred=s>=t; tp=int(sum(pred&(y==1))); fp=int(sum(pred&(y==0))); fn=int(sum(~pred&(y==1))); tn=int(sum(~pred&(y==0)))
    return dict(auroc=roc_auc_score(y,s),ap=average_precision_score(y,s),threshold=t,accuracy=(tp+tn)/len(y),precision=tp/(tp+fp) if tp+fp else 0,recall=tp/(tp+fn),f1=2*tp/(2*tp+fp+fn),tp=tp,fp=fp,fn=fn,tn=tn)
def cdf(s,knots):
    k=np.sort(knots); return (np.searchsorted(k,s,'left')+np.searchsorted(k,s,'right'))/(2*len(k))
def main(baseline=False):
    d=read(Path('E:/异常检测/数据划分/验证集.csv')); y=d['二分类标签'].astype(int).to_numpy()
    assert len(y)==755 and sum(y)==199 and d['钱币编号'].is_unique
    scores={'A1-384':aligned(R/'A1/结果/val_predictions.csv',d)}
    sources=[R/'A1/代码/train.py',R/'A1/模型/config.json',R/'A1/模型/best.pt',R/'A1/结果/val_predictions.csv',R/'A0/模型/convnext_tiny-983f1562.pth']
    sources += [Path('E:/异常检测/数据划分')/n for n in ['训练集.csv','验证集.csv','测试集.csv']]
    audit=ROOT/'审计/input_hashes.json'
    hashes={str(p):sha(p) for p in sources}
    if audit.exists(): assert json.loads(audit.read_text(encoding='utf8'))==hashes
    else: save(audit,hashes)
    original=(R/'A1/代码/train.py').read_text(encoding='utf8').strip()
    modified=(ROOT/'代码/train.py').read_text(encoding='utf8').strip()
    assert original.replace('T.Resize((384,384)','T.Resize((512,512)')==modified
    if not baseline: scores['A8-512']=aligned(OUT/'val_predictions.csv',d)
    rows=[]; groups=[]; thresholds={}; predictions=d.copy()
    def add(name,s):
        thresholds[name]={}; predictions[name]=s
        for point in ['0.5','recall154','maxF1']:
            t=threshold(y,s,point); thresholds[name][point]=t; m=measure(y,s,t); rows.append(dict(model=name,point=point,**m))
            pred=s>=t
            for g in ['AU','XF','92']:
                mask=(y==0)&d['原始评级'].str.startswith(g).to_numpy() if g!='92' else (y==1)&d['异常代码'].str.contains('92').to_numpy()
                n=int(sum(mask)); errors=int(sum((pred!=y)&mask)); groups.append(dict(model=name,point=point,group=g,n=n,errors=errors,error_rate=errors/n if n else None))
    for n,s in scores.items(): add(n,s)
    a1=measure(y,scores['A1-384'],threshold(y,scores['A1-384'],'recall154'))
    assert abs(a1['auroc']-.8537019630526734)<1e-10 and a1['fp']==102 and a1['fn']==45
    if baseline:
        pd.DataFrame(rows).to_csv(OUT/'baseline_metrics.csv',index=False,encoding='utf-8-sig'); return
    a8=measure(y,scores['A8-512'],threshold(y,scores['A8-512'],'recall154'))
    benefit=a8['auroc']>a1['auroc'] or (a8['tp']==154 and a8['fp']<a1['fp'])
    for p,h in json.loads((ROOT/'审计/fusion_input_hashes.json').read_text(encoding='utf-8-sig')).items(): assert sha(Path(p))==h
    fusion_config=json.loads((R/'A1/融合对照/A0_A1集成/frozen_ensemble_config.json').read_text(encoding='utf8'))
    local_config=json.loads((R/'A5/frozen_validation_config.json').read_text(encoding='utf8'))
    old=aligned(R/'A1/融合对照/A0_A1集成/val_predictions.csv',d,'ensemble_score')
    a0=aligned(R/'A0/结果/val_predictions.csv',d)
    a5=aligned(R/'A5/结果/local_scores.csv',d,'A5b')
    alpha=fusion_config['alpha']; w=local_config['fusion']['A0+A1+A5b']['weight']; knots=local_config['cdf_knots']
    assert np.allclose(old,alpha*a0+(1-alpha)*scores['A1-384'],rtol=0,atol=1e-14)
    old3=(1-w)*cdf(old,knots['A0+A1'])+w*cdf(a5,knots['A5b'])
    ref=measure(y,old3,threshold(y,old3,'recall154')); assert abs(ref['auroc']-.8701104443078703)<1e-10 and ref['fp']==92 and ref['fn']==45
    scores.update({'A0+A1':old,'A0+A1+A5b':old3}); add('A0+A1',old); add('A0+A1+A5b',old3)
    if benefit:
        new=alpha*a0+(1-alpha)*scores['A8-512']; new3=(1-w)*cdf(new,knots['A0+A1'])+w*cdf(a5,knots['A5b'])
        scores.update({'A0+A8':new,'A0+A8+A5b':new3}); add('A0+A8',new); add('A0+A8+A5b',new3)
    save(OUT/'frozen_fusion.json',dict(alpha=alpha,a5b_weight=w,cdf='original A5 saved knots retained; no recalibration or weight search',benefit_gate='A8 AUROC improves OR exact TP154 FP decreases',gate_passed=benefit))
    transitions=[]
    for point in ['recall154','maxF1','0.5']:
        p1=scores['A1-384']>=thresholds['A1-384'][point]; p8=scores['A8-512']>=thresholds['A8-512'][point]
        for i in np.flatnonzero(p1!=p8): transitions.append(dict(d.iloc[i],point=point,a1_score=scores['A1-384'][i],a8_score=scores['A8-512'][i],change='corrected' if p8[i]==y[i] else 'introduced',kind='FP' if y[i]==0 else 'FN'))
    pd.DataFrame(transitions).to_csv(OUT/'error_transitions.csv',index=False,encoding='utf-8-sig')
    if benefit:
        changes=[]
        for oldn,newn in [('A0+A1','A0+A8'),('A0+A1+A5b','A0+A8+A5b')]:
            for point in ['recall154','maxF1']:
                op=scores[oldn]>=thresholds[oldn][point]; npred=scores[newn]>=thresholds[newn][point]
                for i in np.flatnonzero(op!=npred): changes.append(dict(d.iloc[i],comparison=oldn+' -> '+newn,point=point,old_score=scores[oldn][i],new_score=scores[newn][i],change='corrected' if npred[i]==y[i] else 'introduced',kind='FP' if y[i]==0 else 'FN'))
        pd.DataFrame(changes).to_csv(OUT/'fusion_error_transitions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(OUT/'metrics_comparison.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(groups).to_csv(OUT/'group_errors.csv',index=False,encoding='utf-8-sig'); predictions.to_csv(OUT/'aligned_val_predictions.csv',index=False,encoding='utf-8-sig'); save(OUT/'thresholds.json',thresholds)
    rng=np.random.default_rng(42); dif=[]
    for _ in range(2000):
        ix=rng.integers(len(y),size=len(y)); dif.append(roc_auc_score(y[ix],scores['A8-512'][ix])-roc_auc_score(y[ix],scores['A1-384'][ix]))
    ci=np.quantile(dif,[.025,.975]); save(OUT/'paired_bootstrap.json',dict(delta=a8['auroc']-a1['auroc'],ci95=ci.tolist(),replicates=2000,note='fixed selected predictions; does not include seed or model selection uncertainty'))
    history=json.loads((OUT/'history.json').read_text(encoding='utf8')); best=json.loads((ROOT/'模型/threshold.json').read_text(encoding='utf8'))
    assert best['best_epoch']==max(history,key=lambda h:h['val_auroc'])['epoch']
    assert abs(best['val_metrics']['auroc']-a8['auroc'])<1e-10
    lines=['# A8 高分辨率双面异常检测实验','',f"完成 {len(history)} 轮，最佳轮次 {best['best_epoch']}。同一验证集755枚（正常556、异常199），A1原预测复用。测试集仅在验证选模和阈值冻结后按A1原流程推理，不参与选择。",'',f"512相对384：AUROC {a1['auroc']:.6f} → {a8['auroc']:.6f}，AP {a1['ap']:.6f} → {a8['ap']:.6f}。主召回实际 TP={a8['tp']}，FP {a1['fp']} → {a8['fp']}，FN {a1['fn']} → {a8['fn']}。",f"配对bootstrap AUROC差值95%区间 [{ci[0]:.6f}, {ci[1]:.6f}]，不包含训练随机种子及反复使用验证集的选择不确定性。",'','| 模型 | 阈值口径 | AUROC | AP | Accuracy | Precision | Recall | F1 | FP | FN |','|---|---|---|---|---|---|---|---|---|---|']
    for r in rows: lines.append('| '+ ' | '.join([r['model'],r['point']]+[f"{r[k]:.6f}" for k in ['auroc','ap','accuracy','precision','recall','f1']]+[str(r['fp']),str(r['fn'])])+' |')
    lines+=['','主召回阈值取异常得分第154高，所有同分样本整体判定，不拆分同分；若实际TP超过154则不是严格相同召回，不能直接宣称FP改善。maxF1并列取最高阈值，沿用A1。','','| 模型 | AU误报 | XF误报 | 92漏检 |','|---|---|---|---|']
    for n in scores:
        g=[next(r for r in groups if r['model']==n and r['point']=='recall154' and r['group']==k) for k in ['AU','XF','92']]; lines.append('| '+n+' | '+' | '.join(f"{r['errors']}/{r['n']}" for r in g)+' |')
    tr=pd.DataFrame(transitions)
    for kind in ['FP','FN']:
        sub=tr[(tr.point=='recall154')&(tr.kind==kind)] if len(tr) else tr
        lines.append(f"主召回下{kind}：修正 {sum(sub.change=='corrected') if len(sub) else 0} 枚，新增 {sum(sub.change=='introduced') if len(sub) else 0} 枚，编号见error_transitions.csv。")
    lines+=['',f"控制：A1代码仅Resize 384→512；全骨干及MLP可训练，原ImageNet权重SHA校验，随机种子42、batch8、FP16、AdamW、调度、早停及选模均不变。使用相同本机RTX5060和原环境、原始分辨率图像缓存，未使用A5d特征。训练总计 {sum(h['epoch_total_seconds'] for h in history)/60:.1f} 分钟，峰值allocated {max(h['allocated_MiB'] for h in history):.0f} MiB。",'', '融合保持0.2*A0+0.8*A8；三级融合保持0.9*原冻结CDF(双面融合)+0.1*原冻结CDF(A5b)。保留原CDF节点，不重新校准或搜索权重。' if benefit else '单模型收益门槛未通过，不运行A8融合；原融合仅作参考。']
    if benefit:
        nf=measure(y,scores['A0+A8+A5b'],thresholds['A0+A8+A5b']['recall154']); replace=nf['auroc']>ref['auroc'] and nf['tp']==154 and nf['fp']<92
        lines.append('融合替换建议：'+('A0+A8+A5b可列为更优候选；建议暂不直接替换现有默认方案，先用独立数据确认这4枚误报的净收益。A0+A8双模型在主召回下更差，不建议替换对应双模型方案。本次未更改既有融合配置。' if replace else '不建议替换原方案；新融合未同时改善AUROC与严格主召回误报。'))
    else: lines.append('融合替换建议：保留原有方案。')
    lines.append('768建议：'+('可以作为后续单独实验候选，但先评估新增算力成本；本次未启动768。' if a8['auroc']>a1['auroc'] and a8['fp']<a1['fp'] and a8['tp']==154 else '暂不优先尝试768：512单模型未改善主召回误报，应先验证512融合的小幅收益；本次未启动768。'))
    old_history=json.loads((R/'A1/结果/history.json').read_text(encoding='utf8'))
    old_time=np.mean([h['epoch_total_seconds'] for h in old_history[1:]])
    new_time=np.mean([h['epoch_total_seconds'] for h in history[1:]])
    lines.append(f'计算成本：排除首轮启动，384平均每轮{old_time:.1f}秒，512为{new_time:.1f}秒，约{new_time/old_time:.2f}倍；实际计时受运行环境波动影响。原缓存抽查48张均为1024×1024，与原图逐字节一致。')
    if ci[0]<=0<=ci[1]: lines.append('收益强度：AUROC差值区间跨0，本次单种子验证不能确认稳定优势；768只宜作为低优先级候选，不能假定分辨率越高越好。')
    if (ROOT/'审计/inference_recovery.json').exists(): lines.append('运行恢复：15轮训练完成后，原finish函数新增数据加载进程导致Windows页面文件不足。保留全部训练产物，仅以num_workers=0重跑最佳权重推理；batch、FP16、模型和阈值规则不变，未重新训练。详情及原始错误日志已保留。')
    (ROOT/'实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save(OUT/'COMPLETE.json',dict(status='complete',epochs=len(history),best_epoch=best['best_epoch'],a1=a1,a8=a8,fusion_run=benefit,source_integrity=True))
    print(json.dumps(dict(a1=a1,a8=a8,fusion_run=benefit),ensure_ascii=False))
if __name__=='__main__': main('--baseline' in sys.argv)
