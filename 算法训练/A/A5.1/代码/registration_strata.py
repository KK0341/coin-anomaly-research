from common import *
from sklearn.metrics import roc_auc_score,average_precision_score
scores=pd.read_csv(OUT/'predictions.csv',dtype={'钱币编号':str});local=read(OUT/'local_scores.csv');threshold=json.loads((OUT/'thresholds.json').read_text());target=str(154/199);q=local[(local.method=='A5c')&(local.seed=='42')].set_index('钱币编号').loc[val['钱币编号']]
count=(q.query_front_aligned.eq('True').astype(int)+q.query_back_aligned.eq('True').astype(int)).values;y=val['二分类标签'].astype(int).values;rows=[]
for name in ['A5b','A5c','A5d','融合_A5b','融合_A5c','融合_A5d']:
 for seed in SEEDS:
  s=scores[(scores.model==name)&(scores.seed==seed)].set_index('钱币编号').loc[val['钱币编号']].score.values;p=s>=threshold[f'{name}_{seed}'][target]
  for num in [0,1,2]:
   mask=count==num;yy=y[mask];rows.append({'model':name,'seed':seed,'aligned_query_sides':num,'n':int(mask.sum()),'normal_n':int((yy==0).sum()),'abnormal_n':int((yy==1).sum()),'auroc':roc_auc_score(yy,s[mask]) if len(set(yy))==2 else None,'ap':average_precision_score(yy,s[mask]) if yy.sum() else None,'fp':int(((yy==0)&p[mask]).sum()),'fn':int(((yy==1)&~p[mask]).sum())})
df=pd.DataFrame(rows);df.to_csv(OUT/'registration_strata.csv',index=False,encoding='utf-8-sig');df.groupby(['model','aligned_query_sides'])[['n','normal_n','abnormal_n','auroc','ap','fp','fn']].mean().to_csv(OUT/'registration_strata_summary.csv',encoding='utf-8-sig');print('STRATA_DONE')
