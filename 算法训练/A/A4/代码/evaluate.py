"""Freeze validation choices before any new test inference; report cached predictions only."""
import sys,json,csv,hashlib,time,importlib.util
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
R=Path(__file__).resolve().parents[1]; B=R.parent;O=R/'结果';NAMES=['A1','A3','A4a','A4b']
def directory(n):return B/n if n in ['A1','A3'] else R/n
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def write(p,rs):
 if not rs:return
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def manifest(split):return read(R/'A4a/结果'/f'{split}_manifest.csv')
def data(n,split):
 rs=manifest(split);ps=read(directory(n)/'结果'/f'{split}_predictions.csv');ix={r['钱币编号']:r for r in ps};assert len(ix)==len(rs)==len(ps)
 ordered=[ix[r['钱币编号']] for r in rs];assert all(int(r['true_label'])==int(s['二分类标签']) for r,s in zip(ordered,rs))
 return np.array([int(r['二分类标签']) for r in rs]),np.array([float(r['coin_score']) for r in ordered]),ordered
def met(y,p,t):
 pred=p>=t;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'auroc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred)),'f1':float(f1_score(y,pred)),'fp':int(fp),'fn':int(fn),'tp':int(tp),'tn':int(tn),'fpr':float(fp/(fp+tn))}
def freeze():
 assert not any((R/n/'结果/test_predictions.csv').exists() for n in ['A4a','A4b'])
 frozen={'created_unix':time.time(),'models':{},'decision':{},'test_policy':'exploratory; frozen once; no test-dependent retraining'}
 protocol=json.loads((R/'protocol.json').read_text(encoding='utf-8'));targets=protocol['matched_recall_targets']
 for n in NAMES:
  d=directory(n);y,p,_=data(n,'val');t=json.loads((d/'模型/threshold.json').read_text(encoding='utf-8'))['threshold']
  matched=[{'target':q,'threshold':float(np.sort(p[y==1])[::-1][int(np.ceil(q*y.sum()-1e-10))-1])} for q in targets]
  frozen['models'][n]={'checkpoint_sha256':sha(d/'模型/best.pt'),'prediction_sha256':sha(d/'结果/val_predictions.csv'),'threshold':t,'val':met(y,p,t),'matched':[{**v,**met(y,p,v['threshold'])} for v in matched]}
 for n,b in [('A4a','A1'),('A4b','A3')]:
  a=frozen['models'][n];base=frozen['models'][b];delta=a['val']['auroc']-base['val']['auroc'];fpdelta=a['matched'][2]['fp']-base['matched'][2]['fp'];promising=delta>=.005 and fpdelta<=0
  frozen['decision'][n]={'baseline':b,'val_auroc_delta':delta,'val_primary_fp_delta':fpdelta,'paired_multiseed_worth_followup':promising,'current_round':'two seed42 core experiments only; no additional search','reason':'meets preregistered validation gate; paired seeds recommended for later confirmation' if promising else 'does not meet preregistered validation gate; no multi-seed expansion this round'}
 save(R/'frozen_evaluation.json',frozen);print(json.dumps(frozen['decision'],ensure_ascii=False))
def infer():
 frozen=json.loads((R/'frozen_evaluation.json').read_text(encoding='utf-8'))
 for n in ['A4a','A4b']:
  d=directory(n);assert sha(d/'模型/best.pt')==frozen['models'][n]['checkpoint_sha256']
  if (d/'结果/test_predictions.csv').exists():continue
  sp=importlib.util.spec_from_file_location(n,d/'代码/train.py');t=importlib.util.module_from_spec(sp);sys.modules[n]=t;sp.loader.exec_module(t)
  save(d/'结果/TEST_INFERENCE_STARTED.json',{'time_unix':time.time(),'frozen_sha256':sha(R/'frozen_evaluation.json')})
  m=t.Dual().cuda();ck=t.torch.load(d/'模型/best.pt',map_location='cpu',weights_only=False);m.load_state_dict(ck['model']);ds=t.Coins('test');v=t.predict(m,t.loader(ds,8,workers=0));t.write_predictions('test',ds,v,frozen['models'][n]['threshold']);del m;t.torch.cuda.empty_cache()
def analyze():
 frozen=json.loads((R/'frozen_evaluation.json').read_text(encoding='utf-8'));comparison=[];matched=[];groups=[];codes=[];aux=[];trans=[];boot=[]
 for split in ['val','test']:
  rs=manifest(split);arr={n:data(n,split)[1] for n in NAMES};y=data('A1',split)[0]
  for n,p in arr.items():
   f=frozen['models'][n];comparison.append({'split':split,'model':n,'threshold':f['threshold'],**met(y,p,f['threshold'])})
   for op,t in [('F1',f['threshold'])]+[(str(v['target']),v['threshold']) for v in f['matched']]:
    if op!='F1':matched.append({'split':split,'model':n,'target':op,'threshold':t,**met(y,p,t)})
    for kind,labels in [('grade',['VF及以下','XF','AU','MS','未监督']),('code',sorted({r['异常代码'] for r in rs if r['二分类标签']=='1'}))]:
     for label in labels:
      mask=np.array([r['grade_group']==label and y[i]==0 if kind=='grade' else r['异常代码']==label and y[i]==1 for i,r in enumerate(rs)]);num=int(mask.sum());errors=int(((p>=t)!=y)[mask].sum())
      groups.append({'split':split,'model':n,'op':op,'kind':kind,'group':label,'n':num,'errors':errors,'rate':errors/num if num else None})
   for code in sorted({r['异常代码'] for r in rs if r['二分类标签']=='1'}):
    mask=np.array([y[i]==0 or r['异常代码']==code for i,r in enumerate(rs)])
    codes.append({'split':split,'model':n,'code':code,'normal_n':int((y[mask]==0).sum()),'abnormal_n':int(y[mask].sum()),'auroc':float(roc_auc_score(y[mask],p[mask])),'ap':float(average_precision_score(y[mask],p[mask]))})
   if n.startswith('A4'):
    _,_,ps=data(n,split);truth=np.array([int(r['reason_id']) for r in rs]);mask=truth>=0;logits=np.array([[float(r[f'reason_logit_{j}']) for j in range(4)] for r in ps]);pred=logits.argmax(1);probs=np.exp(logits-logits.max(1,keepdims=True));probs/=probs.sum(1,keepdims=True)
    cr=classification_report(truth[mask],pred[mask],labels=[0,1,2,3],target_names=['正常','92','97','98'],output_dict=True,zero_division=0)
    save(O/f'{n}_{split}_reason_diagnostic.json',{'classification_report':cr,'confusion_matrix':confusion_matrix(truth[mask],pred[mask],labels=[0,1,2,3]).tolist(),'eligible_n':int(mask.sum()),'abnormal_only_accuracy':float((pred[truth>0]==truth[truth>0]).mean()),'abnormal_only_conditional_accuracy':float((logits[truth>0,1:].argmax(1)+1==truth[truth>0]).mean()),'abnormal_only_majority_baseline':float((truth[truth>0]==1).mean()),'majority_accuracy':float((truth[mask]==0).mean()),'majority_macro_f1':float(f1_score(truth[mask],np.zeros(mask.sum()),labels=[0,1,2,3],average='macro')),'reason_ovr_auroc':{str(k):float(roc_auc_score(truth[mask]==k,probs[mask,k])) for k in range(4)}})
    aux.append({'split':split,'model':n,'accuracy':cr['accuracy'],'macro_f1':cr['macro avg']['f1-score'],'balanced_accuracy':float(np.mean([cr[k]['recall'] for k in ['正常','92','97','98']])),'reason92_recall':cr['92']['recall'],'reason97_recall':cr['97']['recall'],'reason98_recall':cr['98']['recall']})
  for n,b in [('A4a','A1'),('A4b','A3'),('A4b','A4a')]:
   before=arr[b]>=frozen['models'][b]['threshold'];after=arr[n]>=frozen['models'][n]['threshold']
   for i,r in enumerate(rs):
    old=before[i]!=y[i];new=after[i]!=y[i];trans.append({'split':split,'model':n,'baseline':b,'id':r['钱币编号'],'label':int(y[i]),'grade':r['grade_group'],'code':r['异常代码'],'before_score':arr[b][i],'after_score':arr[n][i],'transition':'corrected' if old and not new else 'introduced' if not old and new else 'both_wrong' if old else 'both_right'})
   rng=np.random.default_rng(20261113);deltas=[];idx0=np.flatnonzero(y==0);idx1=np.flatnonzero(y==1)
   for _ in range(2000):
    ix=np.r_[rng.choice(idx0,len(idx0)),rng.choice(idx1,len(idx1))];deltas.append(roc_auc_score(y[ix],arr[n][ix])-roc_auc_score(y[ix],arr[b][ix]))
   boot.append({'split':split,'model':n,'baseline':b,'auroc_delta':roc_auc_score(y,arr[n])-roc_auc_score(y,arr[b]),'ci025':float(np.quantile(deltas,.025)),'ci975':float(np.quantile(deltas,.975)),'note':'paired stratified coin bootstrap; conditional on trained models; not seed uncertainty'})
 for filename,rows in [('comparison',comparison),('matched_recall',matched),('group_errors',groups),('per_code_auroc',codes),('reason_diagnostics',aux),('error_transitions',trans),('paired_bootstrap',boot)]:write(O/f'{filename}.csv',rows)
 save(O/'evaluation.json',{'comparison':comparison,'matched_recall':matched,'groups':groups,'codes':codes,'aux':aux,'bootstrap':boot})
 print(json.dumps(comparison,ensure_ascii=False))
if __name__=='__main__':{'freeze':freeze,'infer':infer,'analyze':analyze}[sys.argv[1]]()
