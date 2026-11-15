from common import *
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=[]
for _,r in val.iterrows():
 cid=r['钱币编号'];z=np.load(OUT/'maps_seed42'/('A5d_'+cid+'.npz'));v=z['valid'];seam=np.isin(v%48,[23,24])|np.isin(v//48,[23,24]);dist=z['distances'];k=math.ceil(len(v)*.05)
 for side in [0,1]:
  top=np.argsort(dist[side])[-k:];rows.append({'钱币编号':cid,'side':side,'label':int(r['二分类标签']),'grade':r['grade_group'],'seam_valid_area_fraction':float(seam.mean()),'max_on_seam':bool(seam[np.argmax(dist[side])]),'top5pct_on_seam_fraction':float(seam[top].mean())})
df=pd.DataFrame(rows);df.to_csv(OUT/'crop_seam_audit.csv',index=False,encoding='utf-8-sig');summary={'seed':42,'definition':'48 grid columns23/24 or rows23/24, one grid cell each side of crop seams','valid_area_fraction':float(df.seam_valid_area_fraction.mean()),'max_on_seam_fraction':float(df.max_on_seam.mean()),'top5pct_on_seam_fraction':float(df.top5pct_on_seam_fraction.mean()),'interpretation':'descriptive location concentration only, not proof of artifact causality or lesion annotation'};save(OUT/'crop_seam_summary.json',summary);print(summary)
plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
m=pd.read_csv(OUT/'matched_summary.csv');fig,ax=plt.subplots(figsize=(8,5))
for name in ['A0+A1','融合_A5b','融合_A5c','融合_A5d']:
 s=m[m.model==name].sort_values('recall_mean');x=s.recall_mean.values;y=s.fp_mean.values;err=s.fp_std.fillna(0).values;ax.plot(x,y,'o-',label=name);ax.fill_between(x,y-err,y+err,alpha=.12)
ax.set(xlabel='验证异常召回率',ylabel='正常误报数（五种子均值±标准差）',title='固定10%融合：不同召回要求下的取舍');ax.legend();fig.tight_layout();fig.savefig(OUT/'可视化/matched_recall.png',dpi=140)
