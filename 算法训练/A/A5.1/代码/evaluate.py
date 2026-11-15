from common import *
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,precision_recall_curve
D=read(OUT/'local_scores.csv');ys=val['二分类标签'].astype(int).values
base=read(R/'A1/融合对照/A0_A1集成/val_predictions.csv').set_index('钱币编号').loc[val['钱币编号']];assert np.array_equal(base.true_label.astype(int),ys)
bp=base.ensemble_score.astype(float).values;br=(rankdata(bp)-.5)/len(bp)
metrics=[];groups=[];predictions=[];cal={'global':sorted(bp.tolist()),'local':{}};thresholds={}
def metric(name,seed,s,point,t):
 p=s>=t;tp=int(((ys==1)&p).sum());fp=int(((ys==0)&p).sum());fn=int(((ys==1)&~p).sum())
 metrics.append({'model':name,'seed':seed,'point':point,'auroc':roc_auc_score(ys,s),'ap':average_precision_score(ys,s),'f1':f1_score(ys,p),'recall':tp/sum(ys),'fp':fp,'fn':fn,'threshold':float(t)})
 for g in ['VF及以下','XF','AU','MS','82','91','92','94','95','97','98']:
  grade=g in ['VF及以下','XF','AU','MS'];mask=(ys==0)&(val.grade_group==g).values if grade else (ys==1)&val['异常代码'].str.contains(g).values
  err=int((p[mask] if grade else ~p[mask]).sum());groups.append({'model':name,'seed':seed,'point':point,'group':g,'n':int(mask.sum()),'errors':err,'kind':'FP' if grade else 'FN'})
def evaluate(name,seed,s):
 prec,rec,ts=precision_recall_curve(ys,s);f=2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-12);t=ts[np.flatnonzero(f==f.max())[-1]];metric(name,seed,s,'maxF1',t);thresholds[f'{name}_{seed}']={'maxF1':float(t)}
 for target in [.7,.75,154/199,.8,.85,.9]:
  t=np.sort(s[ys==1])[::-1][math.ceil(target*sum(ys))-1];metric(name,seed,s,str(target),t);thresholds[f'{name}_{seed}'][str(target)]=float(t)
 for i,cid in enumerate(val['钱币编号']):predictions.append({'钱币编号':cid,'model':name,'seed':seed,'score':float(s[i])})
evaluate('A0+A1',0,bp)
for method in ['A5a','A5b','A5c','A5d']:
 for seed in SEEDS:
  sub=D[(D.method==method)&(D.seed==str(seed))].set_index('钱币编号').loc[val['钱币编号']];s=sub.score.astype(float).values;assert len(s)==755
  cal['local'][f'{method}_{seed}']=sorted(s.tolist());evaluate(method,seed,s);fused=.9*br+.1*(rankdata(s)-.5)/len(s);evaluate('融合_'+method,seed,fused)
m=pd.DataFrame(metrics);g=pd.DataFrame(groups);pr=pd.DataFrame(predictions)
for name,df in [('metrics_per_seed',m),('groups_per_seed',g),('predictions',pr)]:df.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
save(O/'calibration.json',cal);save(OUT/'thresholds.json',thresholds)
summary=m[m.point=='maxF1'].groupby('model')[['auroc','ap','f1']].agg(['mean','std','min','max']);summary.columns=['_'.join(c) for c in summary.columns];summary.to_csv(OUT/'summary.csv',encoding='utf-8-sig')
matched=m[m.point!='maxF1'].groupby(['model','point'])[['fp','fn','recall']].agg(['mean','std','min','max']);matched.columns=['_'.join(c) for c in matched.columns];matched.to_csv(OUT/'matched_summary.csv',encoding='utf-8-sig')
gs=g.groupby(['model','point','group'])['errors'].agg(['mean','std','min','max']).reset_index();gs.to_csv(OUT/'group_summary.csv',index=False,encoding='utf-8-sig')
# Paired seed deltas; seed dispersion does not represent sampling uncertainty of the validation set.
rows=[]
for a,b in [('A5b','A5a'),('A5c','A5b'),('A5d','A5c'),('融合_A5b','融合_A5a'),('融合_A5c','融合_A5b'),('融合_A5d','融合_A5c')]:
 for seed in SEEDS:
  aa=m[(m.model==a)&(m.seed==seed)&(m.point=='maxF1')].iloc[0];bb=m[(m.model==b)&(m.seed==seed)&(m.point=='maxF1')].iloc[0];rows.append({'a':a,'b':b,'seed':seed,'auroc_delta':aa.auroc-bb.auroc,'ap_delta':aa.ap-bb.ap})
pd.DataFrame(rows).to_csv(OUT/'paired_seed_deltas.csv',index=False,encoding='utf-8-sig')
reg=[]
for prefix,cids in json.loads((OUT/'extraction_tasks.json').read_text(encoding='utf8')).items():
 for cid in cids:
  recs=json.loads((C/key(prefix)/(cid+'.json')).read_text(encoding='utf8'))
  for side,r in enumerate(recs):reg.append({'钱币编号':cid,'prefix':prefix,'side':side,**{k:v for k,v in r.items() if k!='matrix'}})
pd.DataFrame(reg).to_csv(OUT/'registration_audit.csv',index=False,encoding='utf-8-sig')
qc=D[D.method=='A5c'].drop_duplicates('钱币编号');rates={'query_side_fallback':float(np.mean(np.concatenate([qc.query_front_aligned!='True',qc.query_back_aligned!='True']))),'pair_side_fallback_by_seed':D[D.method=='A5c'].groupby('seed').pair_side_fallback_fraction.apply(lambda s:s.astype(float).mean()).to_dict(),'all_image_side_fallback':float(np.mean([not r['accepted'] for r in reg]))};save(OUT/'registration_summary.json',rates)
# Historical baseline exact replay, normal-bank membership, frozen inputs, spatial constraints.
oldscore=read(R/'A5/结果/local_scores.csv').set_index('钱币编号').loc[val['钱币编号']]
check=D[(D.method=='A5b')&(D.seed=='42')].set_index('钱币编号').loc[val['钱币编号']].score.astype(float).values
maxerr=float(np.max(np.abs(check-oldscore.A5b.astype(float).values)));assert maxerr<1e-6,maxerr
assert set(train['钱币编号']).isdisjoint(val['钱币编号'])
plan=json.loads((O/'reference_plans.json').read_text(encoding='utf8'));assert all(len(ids)==32 and len(set(ids))==32 and normal.iloc[ids]['二分类标签'].eq('0').all() for p in plan.values() for ids in p['seeds'].values())
hs=json.loads((OUT/'input_hashes.json').read_text());assert all(sha(Path(p))==h for p,h in hs.items())
assert np.isfinite(pr.score).all();assert len(pr)==755*41
# Verify actual NN positions for reliable query/reference pair, and score-from-map consistency for all seed42 queries.
spatial_checks=0
for method in ['A5a','A5b','A5c','A5d']:
 n=48 if method=='A5d' else 24
 for _,r in val.iterrows():
  cid=r['钱币编号'];z=np.load(OUT/'maps_seed42'/(method+'_'+cid+'.npz'));ds=z['distances'];score=np.sort(ds,axis=1)[:,-math.ceil(ds.shape[1]*.05):].mean(1).max();expected=float(D[(D.method==method)&(D.seed=='42')&(D['钱币编号']==cid)].score.iloc[0]);assert abs(score-expected)<1e-6
  if method not in ['A5c','A5d']:continue
  prefix=prefix_for(r);folder=C/key(prefix);qr=json.loads((folder/(cid+'.json')).read_text());valid=z['valid'];refs=z['ref_indices'];rp=z['ref_patches']
  for side in [0,1]:
   if not qr[side]['accepted']:continue
   for rid in np.unique(refs[side]):
    refid=normal.iloc[int(rid)]['钱币编号'];rr=json.loads((folder/(refid+'.json')).read_text())
    if not rr[side]['accepted']:continue
    take=refs[side]==rid;dx=np.abs(valid[take]%n-rp[side,take]%n);dy=np.abs(valid[take]//n-rp[side,take]//n);assert np.maximum(dx,dy).max()<=(2 if n==48 else 1);spatial_checks+=int(take.sum())
save(OUT/'verification.json',{'passed':True,'A5b_seed42_max_score_error':maxerr,'spatially_constrained_matches_checked':spatial_checks,'historical_hashes_unchanged':True,'train_val_disjoint':True,'refs_32_normal_train_only':True,'map_score_recomputed':True,'test_access':False})
print(summary.to_string());print(rates);print('EVALUATION_DONE',flush=True)
