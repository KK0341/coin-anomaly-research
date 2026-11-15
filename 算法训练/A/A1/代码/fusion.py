import train as t
import numpy as np
import csv,json,hashlib
from pathlib import Path
F=t.ROOT/'融合对照'; A0=t.ROOT.parent/'A0'
def read(p):
    with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def scores(rows,kind,w):
    p0=np.array([float(r['front_score']) for r in rows]); p1=np.array([float(r['back_score']) for r in rows])
    if kind=='max':return np.maximum(p0,p1)
    if kind=='front_back':return w*p0+(1-w)*p1
    return w*np.maximum(p0,p1)+(1-w)*(p0+p1)/2

def groups(rows,p,threshold):
    out=[]
    for g in ['AU','XF','MS','92','97','98']:
        ids=[i for i,r in enumerate(rows) if (r['原始评级'].startswith(g) and int(r['true_label'])==0) ] if g in ['AU','XF','MS'] else [i for i,r in enumerate(rows) if g in r['异常代码'] and int(r['true_label'])==1]
        n=sum(int(p[i]>=threshold)!=int(rows[i]['true_label']) for i in ids)
        out.append(dict(group=g,total=len(ids),errors=n,error_rate=n/len(ids) if ids else None))
    return out

def main():
    val=read(A0/'结果/val_predictions.csv')
    assert [r['钱币编号'] for r in val]==[r['钱币编号'] for r in t.rows('val')]
    y=np.array([int(r['true_label']) for r in val]); search=[]; selected={}
    original=json.loads((A0/'模型/threshold.json').read_text(encoding='utf-8'))['threshold']
    for kind in ['max','front_back','max_mean']:
        candidates=[]
        for w in ([1.] if kind=='max' else np.arange(11)/10):
            p=scores(val,kind,w); m=t.metrics(y,p); r=dict(method=kind,weight=float(w),val_auroc=m['auroc'],val_ap=m['ap']); candidates.append(r); search.append(r)
        # AUROC priority; AP breaks exact AUROC ties; then smaller weight.
        best=max(candidates,key=lambda r:(r['val_auroc'],r['val_ap'],-r['weight']))
        p=scores(val,kind,best['weight']); prec,rec,ts=t.precision_recall_curve(y,p); fs=2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-12)
        th=original if kind=='max' else float(ts[np.flatnonzero(fs==fs.max())[-1]])
        selected[kind]={**best,'threshold':th,'val_metrics':t.metrics(y,p,th)}
    overall=max(selected,key=lambda k:(selected[k]['val_auroc'],selected[k]['val_ap']))
    frozen={'selection':'validation AUROC, AP breaks exact ties, then smaller weight; threshold max val F1, highest threshold tie-break; original A0 threshold unchanged','overall_selected':overall,'methods':selected,'source_sha256':{s:hashlib.sha256((A0/f'结果/{s}_predictions.csv').read_bytes()).hexdigest() for s in ['val','test']}}
    t.write_csv(F/'validation_weight_search.csv',search); t.save_json(F/'frozen_validation_selection.json',frozen)
    # Test is read only after selection is persisted. Each predeclared family evaluated once.
    test=read(A0/'结果/test_predictions.csv'); assert [r['钱币编号'] for r in test]==[r['钱币编号'] for r in t.rows('test')]
    ty=np.array([int(r['true_label']) for r in test]); results={}; table=[]
    for kind,sel in selected.items():
        p=scores(test,kind,sel['weight']); th=sel['threshold']; m=t.metrics(ty,p,th); results[kind]={**sel,'test_metrics':m,'groups':groups(test,p,th)}
        table.append(dict(method=kind,weight=sel['weight'],threshold=th,**{k:v for k,v in m.items() if k!='confusion_matrix'}))
        t.write_csv(F/f'{kind}_test_predictions.csv',[{**r,'fusion_score':float(p[i]),'fusion_pred':int(p[i]>=th),'fusion_threshold':th} for i,r in enumerate(test)])
    original_metrics=json.loads((A0/'结果/test_metrics.json').read_text(encoding='utf-8'))
    for k in ['accuracy','precision','recall','f1','auroc','ap']:assert abs(results['max']['test_metrics'][k]-original_metrics[k])<1e-10
    t.save_json(F/'results.json',results); t.write_csv(F/'test_comparison.csv',table)
    print(json.dumps(results,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
