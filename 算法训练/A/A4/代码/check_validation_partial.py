import csv,json,math
from pathlib import Path
B=Path(r'E:\异常检测\算法训练');R=B/'A4';out=[]
for n,d in [('A1',B/'A1'),('A3',B/'A3'),('A4a',R/'A4a')]:
 with (d/'结果/val_predictions.csv').open(encoding='utf-8-sig',newline='') as f:rs=list(csv.DictReader(f))
 pos=sorted([float(r['coin_score']) for r in rs if r['true_label']=='1'],reverse=True)
 for target in [.7,.75,154/199,.8,.85,.9]:
  th=pos[math.ceil(target*len(pos)-1e-10)-1];fp=sum(r['true_label']=='0' and float(r['coin_score'])>=th for r in rs);tp=sum(r['true_label']=='1' and float(r['coin_score'])>=th for r in rs);out.append({'model':n,'target':target,'fp':fp,'recall':tp/len(pos),'threshold':th})
print(json.dumps(out,ensure_ascii=False))
