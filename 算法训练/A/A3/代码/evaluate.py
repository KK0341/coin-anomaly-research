"""Validation selection first; test is a separate explicit stage. No test search."""
import argparse,csv,json,hashlib,time
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_curve,roc_curve
import train as t
R=t.ROOT; O=t.OUT; F=R/'融合'; B=R.parent
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(name,split):
 path=B/name/'结果'/f'{split}_predictions.csv' if name!='A0+A1' else B/'A1/融合对照/A0_A1集成'/f'{split}_predictions.csv'
 ds=t.rows(split);rs=read(path); ix={r['钱币编号']:r for r in rs};assert len(ix)==len(rs)==len(ds)
 ys=np.array([int(r['二分类标签']) for r in ds]);ordered=[ix[r['钱币编号']] for r in ds]
 assert all(int(r['true_label'])==y for r,y in zip(ordered,ys))
 field=('ensemble_score' if split=='val' else 'ensemble_score') if name=='A0+A1' else 'coin_score'
 return ys,np.array([float(r[field]) for r in ordered])
def th(y,p):
 pr,re,ts=precision_recall_curve(y,p);f=2*pr[:-1]*re[:-1]/(pr[:-1]+re[:-1]+1e-12);return float(ts[np.flatnonzero(f==f.max())[-1]])
def met(y,p,threshold):
 r=t.metrics(y,p,threshold);tn,fp,fn,tp=np.array(r.pop('confusion_matrix')).ravel();return {**r,'tp':int(tp),'fp':int(fp),'fn':int(fn),'tn':int(tn)}
def base_th(name):
 p=B/name/'模型/threshold.json' if name!='A0+A1' else B/'A1/融合对照/A0_A1集成/frozen_ensemble_config.json'
 return json.loads(p.read_text(encoding='utf-8'))['threshold']
def validation():
 assert not (O/'test_predictions.csv').exists()
 ys,p3=load('A3','val');_,p0=load('A0','val');_,p1=load('A1','val')
 search=[]
 for a in [0,.1,.2,.3,.5,1.]:
  p=a*p0+(1-a)*p3;threshold=th(ys,p);search.append({'alpha':a,'threshold':threshold,**met(ys,p,threshold)})
 t.write_csv(F/'validation_search.csv',search);best=max(search,key=lambda r:(r['auroc'],r['ap'],-r['alpha']))
 recall_target=met(ys,p1,base_th('A1'))['recall'];targets=sorted(set([.70,.75,recall_target,.80,.85,.90]));matched=[]
 for name,p in [('A1',p1),('A3',p3)]:
  for target in targets:
   # Highest validation threshold achieving required recall minimizes FP.
   threshold=float(np.sort(p[ys==1])[::-1][int(np.ceil(target*sum(ys)-1e-10))-1])
   matched.append({'model':name,'target':target,'threshold':threshold,**met(ys,p,threshold)})
 t.write_csv(O/'matched_recall_validation.csv',matched)
 frozen={'time_unix':time.time(),'grade_lambda':.2,'checkpoint_sha256':hashlib.sha256((t.MODEL/'best.pt').read_bytes()).hexdigest(),'A3_threshold':base_th('A3'),'ensemble':best,'alpha_grid':[0,.1,.2,.3,.5,1.],'matched_recall':matched,'primary_recall_target':recall_target,'selection':'validation AUROC/AP/smaller alpha; max validation F1 threshold','decision':'single lambda run retained; fusion is a validation-selected complementarity control; no test-driven adjustment'}
 t.save_json(t.MODEL/'frozen_evaluation.json',frozen)
 print(json.dumps(frozen,ensure_ascii=False))
def test():
 frozen=json.loads((t.MODEL/'frozen_evaluation.json').read_text(encoding='utf-8'));assert hashlib.sha256((t.MODEL/'best.pt').read_bytes()).hexdigest()==frozen['checkpoint_sha256']
 if not (O/'test_predictions.csv').exists():
  m=t.Dual().cuda();ck=t.torch.load(t.MODEL/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);ds=t.Coins('test');values=t.predict(m,t.loader(ds,8,workers=0));t.write_predictions('test',ds,values,frozen['A3_threshold'])
  grade=np.array([int(r['grade_id']) for r in ds.rows]);mask=grade>=0;t.save_json(O/'grade_test.json',{'n':int(mask.sum()),'accuracy':float((values[4].argmax(1)[mask]==grade[mask]).mean())})
 results=[];groups=[];trans=[];matched=[]
 for split in ['val','test']:
  ds=t.rows(split);arrays={n:load(n,split)[1] for n in ['A0','A1','A0+A1','A2','A3']};y=load('A3',split)[0]
  a=frozen['ensemble']['alpha'];arrays['A0+A3']=a*arrays['A0']+(1-a)*arrays['A3'];thresholds={n:base_th(n) for n in arrays if n!='A0+A3'};thresholds['A0+A3']=frozen['ensemble']['threshold']
  preds={n:p>=thresholds[n] for n,p in arrays.items()}
  for n,p in arrays.items():
   results.append({'split':split,'model':n,'threshold':thresholds[n],**met(y,p,thresholds[n])})
   for g in ['VF及以下','XF','AU','MS','未监督']:
    mask=np.array([r['grade_group']==g and int(r['二分类标签'])==0 for r in ds]);count=int(mask.sum());fp=int(preds[n][mask].sum());groups.append({'split':split,'model':n,'grade':g,'n':count,'fp':fp,'fpr':fp/count if count else None})
  for n in ['A3','A0+A3']:
   for i,r in enumerate(ds):
    before=preds['A1'][i]!=y[i];after=preds[n][i]!=y[i]
    trans.append({'split':split,'model':n,'钱币编号':r['钱币编号'],'原始评级':r['原始评级'],'异常代码':r['异常代码'],'grade':r['grade_group'],'label':int(y[i]),'A1_score':float(arrays['A1'][i]),'new_score':float(arrays[n][i]),'transition':'corrected' if before and not after else 'introduced' if not before and after else 'both_wrong' if before else 'both_right'})
  for r in frozen['matched_recall']:
   matched.append({'split':split,'model':r['model'],'target':r['target'],'threshold':r['threshold'],**met(y,arrays[r['model']],r['threshold'])})
  t.write_csv(F/f'{split}_predictions.csv',[{**r,'score':float(arrays['A0+A3'][i]),'pred_label':int(preds['A0+A3'][i]),'threshold':thresholds['A0+A3']} for i,r in enumerate(ds)])
 t.write_csv(O/'comparison.csv',results);t.write_csv(O/'normal_grade_fp.csv',groups);t.write_csv(O/'error_transitions.csv',trans);t.write_csv(O/'matched_recall.csv',matched)
 t.save_json(O/'evaluation.json',{'comparison':results,'normal_grade_fp':groups,'matched_recall':matched})
 print(json.dumps(results,ensure_ascii=False))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['validation','test']);args=ap.parse_args();validation() if args.stage=='validation' else test()
