import csv,json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,balanced_accuracy_score,confusion_matrix
R=Path(r'E:\异常检测\算法训练\A4');out=[]
for n in ['A4a','A4b']:
 for split in ['val','test']:
  with (R/n/'结果'/f'{split}_predictions.csv').open(encoding='utf-8-sig',newline='') as f:rs=[r for r in csv.DictReader(f) if int(r['reason_id'])>0]
  y=np.array([int(r['reason_id']) for r in rs]);z=np.array([[float(r[f'reason_logit_{j}']) for j in [1,2,3]] for r in rs]);p=z.argmax(1)+1
  out.append({'model':n,'split':split,'n':len(rs),'balanced_accuracy':balanced_accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro'),'majority_macro_f1':f1_score(y,np.ones(len(y)),average='macro'),'confusion_matrix':confusion_matrix(y,p,labels=[1,2,3]).tolist()})
(R/'结果/reason_anomaly_only.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out,ensure_ascii=False))
