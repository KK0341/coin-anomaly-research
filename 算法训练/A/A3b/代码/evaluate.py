"""Freeze using validation only, then one test inference and descriptive analysis."""
import sys,json,csv,hashlib,time
import numpy as np
import train as t
R=t.ROOT;O=t.OUT;M=t.MODEL;B=R.parent;NAMES=['A1','A3','A3b']
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def base(n):return json.loads((B/n/'模型/threshold.json').read_text(encoding='utf-8'))['threshold']
def load(n,s):
 ds=t.rows(s);rs=read(B/n/'结果'/f'{s}_predictions.csv');ix={r['钱币编号']:r for r in rs};assert len(ix)==len(rs)==len(ds)
 ordered=[ix[r['钱币编号']] for r in ds];y=np.array([int(r['二分类标签']) for r in ds]);assert all(int(r['true_label'])==a for r,a in zip(ordered,y))
 return y,np.array([float(r['coin_score']) for r in ordered])
def met(y,p,th):
 d=t.metrics(y,p,th);tn,fp,fn,tp=np.array(d.pop('confusion_matrix')).ravel();return {**d,'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
def freeze():
 assert not (O/'test_predictions.csv').exists()
 assert not (M/'frozen_evaluation.json').exists()
 y,p=load('A1','val');primary=met(y,p,base('A1'))['recall'];targets=sorted(set([.70,.75,.80,.85,.90,primary]));matched=[];comparison=[]
 for n in NAMES:
  y,p=load(n,'val');comparison.append({'model':n,'threshold':base(n),**met(y,p,base(n))})
  for target in targets:
   th=float(np.sort(p[y==1])[::-1][int(np.ceil(target*sum(y)-1e-10))-1]);matched.append({'model':n,'target':target,'threshold':th,**met(y,p,th)})
 t.write_csv(O/'matched_recall_validation.csv',matched)
 f={'time_unix':time.time(),'checkpoint_sha256':sha(M/'best.pt'),'thresholds':{n:base(n) for n in NAMES},'primary_recall_target':primary,'matched_recall':matched,'validation_comparison':comparison,'extra_seed_decision':'complete seed=42 first; no additional runs or parameter changes in this experiment','test_status':'exploratory frozen evaluation; prior test errors informed hypothesis','code_sha256':{p.name:sha(p) for p in (R/'代码').glob('*.py')},'manifest_sha256':{s:sha(O/f'{s}_manifest.csv') for s in ['train','val','test']}}
 t.save_json(M/'frozen_evaluation.json',f);print(json.dumps(f,ensure_ascii=False),flush=True)
def evaluate():
 f=json.loads((M/'frozen_evaluation.json').read_text(encoding='utf-8'));assert sha(M/'best.pt')==f['checkpoint_sha256']
 for s,h in f['manifest_sha256'].items():assert sha(O/f'{s}_manifest.csv')==h
 if not (O/'test_predictions.csv').exists():
  assert not (O/'TEST_INFERENCE_STARTED.json').exists(),'Test inference already attempted; inspect before retry'
  t.save_json(O/'TEST_INFERENCE_STARTED.json',{'time_unix':time.time(),'checkpoint_sha256':f['checkpoint_sha256']})
  m=t.Dual().cuda();ck=t.torch.load(M/'best.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);ds=t.Coins('test');v=t.predict(m,t.loader(ds,8,workers=0));t.write_predictions('test',ds,v,f['thresholds']['A3b']);del m
 results=[];matched=[];groups=[];trans=[];known=[]
 for s in ['val','test']:
  ds=t.rows(s);y=load('A3b',s)[0];ps={n:load(n,s)[1] for n in NAMES};pred={n:ps[n]>=f['thresholds'][n] for n in NAMES}
  for n in NAMES:results.append({'split':s,'model':n,'threshold':f['thresholds'][n],**met(y,ps[n],f['thresholds'][n])})
  ops=[{'model':n,'target':'validation_F1','threshold':f['thresholds'][n]} for n in NAMES]+f['matched_recall']
  for op in ops:
   n=op['model'];p=ps[n]>=op['threshold']
   if op['target']!='validation_F1':matched.append({'split':s,**{k:op[k] for k in ['model','target','threshold']},**met(y,ps[n],op['threshold'])})
   for kind,values in [('normal_grade',['VF及以下','XF','AU','MS','未监督']),('abnormal_code',sorted(set(r['异常代码'] for r in ds if r['二分类标签']=='1')))]:
    for g in values:
     mask=np.array([((int(r['二分类标签'])==0 and r['grade_group']==g) if kind=='normal_grade' else (int(r['二分类标签'])==1 and r['异常代码']==g)) for r in ds]);count=int(mask.sum());errors=int((p[mask]!=y[mask]).sum())
     groups.append({'split':s,'model':n,'operating_point':op['target'],'kind':kind,'group':g,'n':count,'errors':errors,'error_rate':errors/count if count else None})
  for before in ['A1','A3']:
   for i,r in enumerate(ds):
    b=bool(pred[before][i]!=y[i]);a=bool(pred['A3b'][i]!=y[i]);trans.append({'split':s,'baseline':before,**r,'baseline_score':float(ps[before][i]),'A3b_score':float(ps['A3b'][i]),'baseline_pred':int(pred[before][i]),'A3b_pred':int(pred['A3b'][i]),'transition':'corrected' if b and not a else 'introduced' if not b and a else 'both_wrong' if b else 'both_right'})
  if s=='test':
   for i,r in enumerate(ds):
    if y[i]==1 and pred['A1'][i] and not pred['A3'][i]:known.append({**r,**{n+'_score':float(ps[n][i]) for n in NAMES},'A3b_recovered':int(pred['A3b'][i])})
 for name,data in [('comparison',results),('matched_recall',matched),('group_errors',groups),('error_transitions',trans),('A3_added_FN16',known)]:t.write_csv(O/f'{name}.csv',data)
 t.save_json(O/'evaluation.json',{'comparison':results,'matched_recall':matched,'groups':groups,'A3_new_FN_count':len(known),'A3_new_FN_recovered':sum(r['A3b_recovered'] for r in known)})
 print(json.dumps(results,ensure_ascii=False),flush=True)
if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else evaluate()
