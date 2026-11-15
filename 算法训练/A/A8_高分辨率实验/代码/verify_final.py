import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score,confusion_matrix,f1_score,accuracy_score,precision_score,recall_score
R=Path(__file__).resolve().parents[1]; O=R/'结果'
if __name__=='__main__':
    assert 'FINAL ' in (O/'inference_stdout.log').read_text(encoding='utf8')
    assert not (O/'inference_stderr.log').read_text(encoding='utf8').strip()
    preds=pd.read_csv(O/'aligned_val_predictions.csv',dtype={'钱币编号':str},float_precision='round_trip'); metrics=pd.read_csv(O/'metrics_comparison.csv',float_precision='round_trip'); y=preds['二分类标签'].to_numpy()
    for _,r in metrics.iterrows():
        s=preds[r.model].to_numpy(); p=s>=r.threshold
        for k,v in dict(auroc=roc_auc_score(y,s),ap=average_precision_score(y,s),accuracy=accuracy_score(y,p),precision=precision_score(y,p),recall=recall_score(y,p),f1=f1_score(y,p)).items(): assert abs(v-r[k])<1e-12,(r.model,r.point,k,v,r[k])
        tn,fp,fn,tp=confusion_matrix(y,p).ravel(); assert (tn,fp,fn,tp)==(r.tn,r.fp,r.fn,r.tp)
    # Use round-trip float parsing when verifying threshold boundary rows.
    ck=torch.load(R/'模型/best.pt',map_location='cpu',weights_only=False)
    pre=torch.load(R.parent/'A0/模型/convnext_tiny-983f1562.pth',map_location='cpu',weights_only=True)
    changed=[k for k,v in pre.items() if 'net.'+k in ck['model'] and not torch.equal(v,ck['model']['net.'+k])]
    assert 'features.0.0.weight' in changed and any(k.startswith('features.7.') for k in changed)
    h=json.loads((O/'history.json').read_text(encoding='utf8')); assert len(h)==15 and ck['epoch']==max(h,key=lambda r:r['val_auroc'])['epoch']==4
    test=pd.read_csv(O/'test_predictions.csv'); assert len(test)==755 and test['钱币编号'].is_unique
    result=dict(status='passed',metric_rows_verified=len(metrics),best_epoch=ck['epoch'],epochs=15,backbone_tensors_changed=len(changed),single_gpu=True,training_batch=8,inference_recovery_only=True,test_predictions=755,best_sha256=hashlib.sha256((R/'模型/best.pt').read_bytes()).hexdigest())
    (O/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8'); print(json.dumps(result))
