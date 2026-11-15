import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
R=Path(r'E:\异常检测\算法训练'); O=R/'A5'; P=O/'结果'
d=pd.read_csv(P/'val_predictions.csv',dtype={'钱币编号':str}); loc=pd.read_csv(P/'local_scores.csv',dtype={'钱币编号':str}); normal=pd.read_csv(P/'normal_bank_manifest.csv',dtype={'钱币编号':str}); hs=json.loads((P/'input_hashes.json').read_text()); ts=json.loads((P/'thresholds.json').read_text()); target=str(154/199)
assert len(d)==755 and len(normal)==747 and normal['二分类标签'].eq(0).all()
assert set(d['钱币编号']).isdisjoint(normal['钱币编号'])
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hs.items())
assert np.isfinite(d[['A5a','A5b']].values).all()
maps=np.load(P/'local_maps.npz'); valid=np.load(P/'valid_patch_indices.npy'); assert maps['distances'].shape==(755,2,2,len(valid)); assert maps['matches'].max()<32*len(valid)
s=maps['distances']; k=int(np.ceil(len(valid)*.05)); computed=np.sort(s,axis=-1)[...,-k:].mean(-1).max(-1); assert np.allclose(computed,d[['A5a','A5b']].values,atol=1e-6)
assert np.allclose(d.loc[loc.fallback_depth.eq(-1),'A5a'],d.loc[loc.fallback_depth.eq(-1),'A5b'])
rows=[]
for name in ['A5a','A5b','A0+A1+A5a','A0+A1+A5b']:
 a=d['A0+A1'].values>=ts['A0+A1'][target]; b=d[name].values>=ts[name][target]; y=d['二分类标签'].values
 for i in range(len(d)):
  rows.append({'钱币编号':d.iloc[i]['钱币编号'],'model':name,'true_label':int(y[i]),'grade':d.iloc[i].grade_group,'code':d.iloc[i]['异常代码'],'baseline_error':bool(a[i]!=y[i]),'new_error':bool(b[i]!=y[i]),'transition':'corrected' if a[i]!=y[i] and b[i]==y[i] else 'introduced' if a[i]==y[i] and b[i]!=y[i] else 'both_wrong' if b[i]!=y[i] else 'both_right'})
pd.DataFrame(rows).to_csv(P/'error_transitions.csv',index=False,encoding='utf-8-sig')
rows=[]
for group,mask in [('exact',loc.fallback_depth.eq(0)),('ancestor',loc.fallback_depth.gt(0)),('global',loc.fallback_depth.lt(0))]:
 for name in ['A5a','A5b','A0+A1']:
  sub=d[mask]; y=sub['二分类标签']; pred=sub[name]>=ts[name][target]; rows.append({'group':group,'model':name,'n':len(sub),'auroc':roc_auc_score(y,sub[name]) if len(set(y))==2 else None,'fp':int(((y==0)&pred).sum()),'fn':int(((y==1)&~pred).sum())})
pd.DataFrame(rows).to_csv(P/'fallback_metrics.csv',index=False,encoding='utf-8-sig')
checks={'passed':True,'train_normal_only':True,'val_train_id_disjoint':True,'historical_input_hashes_unchanged':True,'local_score_recomputed_from_maps':True,'unknown_fallback_a_equals_b':True,'new_training':False,'test_access_by_A5':False,'feature_files':len(list((O/'特征缓存').glob('*.npy'))),'cache_bytes':sum(p.stat().st_size for p in (O/'特征缓存').glob('*.npy'))}
(P/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8'); print(checks)
