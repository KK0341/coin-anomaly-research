import json,math,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve,f1_score,precision_score,recall_score
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
R=Path(r'E:\异常检测\算法训练'); O=R/'A5'; OUT=O/'结果'
plt.rcParams['font.sans-serif']=['Microsoft YaHei']; plt.rcParams['axes.unicode_minus']=False
def read(p):return pd.read_csv(p,dtype=str).fillna('')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def table(df):
 return '| '+' | '.join(df.columns)+' |\n| '+' | '.join(['---']*len(df.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None))
d=read(OUT/'val_manifest.csv'); local=read(OUT/'local_scores.csv'); assert d['钱币编号'].tolist()==local['钱币编号'].tolist(); y=d['二分类标签'].astype(int).values
scores={}
for a in ['A0','A1','A3']:
 p=read(R/f'{a}/结果/val_predictions.csv').set_index('钱币编号').loc[d['钱币编号']]; assert np.array_equal(p.true_label.astype(int),y); scores[a]=p.coin_score.astype(float).values
p=read(R/'A1/融合对照/A0_A1集成/val_predictions.csv').set_index('钱币编号').loc[d['钱币编号']]; assert np.array_equal(p.true_label.astype(int),y); scores['A0+A1']=p.ensemble_score.astype(float).values
for a in ['A5a','A5b']: scores[a]=local[a].astype(float).values
search=[]; chosen={}; calibration={}
for base in ['A1','A0+A1','A3']:
 for a in ['A5a','A5b']:
  # Fixed validation empirical CDF; save knots for prospective data.
  r1=(rankdata(scores[base],method='average')-.5)/len(y); r2=(rankdata(scores[a],method='average')-.5)/len(y)
  options=[]
  for w in [0,.1,.25,.5,.75,1]:
   s=(1-w)*r1+w*r2; row={'base':base,'local':a,'weight':w,'auroc':roc_auc_score(y,s),'ap':average_precision_score(y,s)}; options.append((row,s));search.append(row)
  win=max(options,key=lambda z:(z[0]['auroc'],z[0]['ap'],-z[0]['weight'])); name=base+'+'+a; scores[name]=win[1]; chosen[name]=win[0]
for name in ['A1','A0+A1','A3','A5a','A5b']:calibration[name]=sorted(scores[name].tolist())
save(O/'frozen_validation_config.json',{'fusion':chosen,'cdf_knots':calibration,'cdf_definition':'(count_less + count_equal/2)/n; same frozen validation knots for prospective inputs','warning':'selected and reported on same repeatedly used validation set; exploratory only'})
pd.DataFrame(search).to_csv(OUT/'fusion_search.csv',index=False,encoding='utf-8-sig')
metrics=[]; matched=[]; groups=[]; thresholds={}; predictions=d.copy()
def measure(name,s,t,point):
 pred=s>=t; row={'model':name,'point':point,'auroc':roc_auc_score(y,s),'ap':average_precision_score(y,s),'f1':f1_score(y,pred),'recall':recall_score(y,pred),'precision':precision_score(y,pred,zero_division=0),'fp':int(((y==0)&pred).sum()),'fn':int(((y==1)&~pred).sum()),'threshold':float(t)}
 for g in ['VF及以下','XF','AU','MS','92','97','98','95','94','91','82']:
  isgrade=g in ['VF及以下','XF','AU','MS']; mask=((y==0)&(d.grade_group==g).values) if isgrade else ((y==1)&d['异常代码'].str.contains(g).values)
  n=int(mask.sum()); err=int((pred[mask] if isgrade else ~pred[mask]).sum()); sub=mask|(y==0)
  groups.append({'model':name,'point':point,'group':g,'n':n,'errors':err,'error_kind':'FP' if isgrade else 'FN','rate':err/n if n else None,'auroc':roc_auc_score(y[sub],s[sub]) if (not isgrade and n) else None})
 return row
for name,s in scores.items():
 prec,rec,ts=precision_recall_curve(y,s); fs=2*prec[:-1]*rec[:-1]/(prec[:-1]+rec[:-1]+1e-12); t=ts[np.flatnonzero(fs==fs.max())[-1]]; thresholds[name]={'maxF1':float(t)}; metrics.append(measure(name,s,t,'maxF1')); predictions[name]=s
 for target in [.7,.75,154/199,.8,.85,.9]:
  t=np.sort(s[y==1])[::-1][math.ceil(target*sum(y))-1]; thresholds[name][str(target)]=float(t); matched.append(measure(name,s,t,str(target)))
metrics=pd.DataFrame(metrics); matched=pd.DataFrame(matched); groups=pd.DataFrame(groups)
for name,df in [('comparison',metrics),('matched_recall',matched),('group_errors',groups),('val_predictions',predictions)]:df.to_csv(OUT/(name+'.csv'),index=False,encoding='utf-8-sig')
save(OUT/'thresholds.json',thresholds)
# Paired bootstrap, fixed scores; selection uncertainty is not included.
boot=[]; rng=np.random.default_rng(42)
for aa,bb in [('A5b','A5a'),('A5b','A1'),('A0+A1+A5b','A0+A1')]:
 dif=[]
 for _ in range(500):
  ix=rng.integers(0,len(y),len(y)); dif.append(roc_auc_score(y[ix],scores[aa][ix])-roc_auc_score(y[ix],scores[bb][ix]))
 boot.append({'model':aa,'baseline':bb,'delta':roc_auc_score(y,scores[aa])-roc_auc_score(y,scores[bb]),'lo':np.quantile(dif,.025),'hi':np.quantile(dif,.975)})
pd.DataFrame(boot).to_csv(OUT/'paired_bootstrap.csv',index=False,encoding='utf-8-sig')
fall=local.fallback_depth.astype(int); summary={'exact':int((fall==0).sum()),'ancestor':int((fall>0).sum()),'global':int((fall<0).sum()),'fallback_rate':float((fall!=0).mean())};save(OUT/'fallback_summary.json',summary)
# Six examples: four pre-experiment difficult samples plus one correction and one new error if available.
ins=json.loads((OUT/'inspection_rows.json').read_text(encoding='utf8')); ids=[r['钱币编号'] for r in ins if r['split']=='val']; base=scores['A0+A1']>=thresholds['A0+A1'][str(154/199)]; new=scores['A0+A1+A5b']>=thresholds['A0+A1+A5b'][str(154/199)]
for mask in [(base!=y)&(new==y),(base==y)&(new!=y)]:
 for i in np.flatnonzero(mask):
  if d.iloc[i]['钱币编号'] not in ids:ids.append(d.iloc[i]['钱币编号']);break
maps=np.load(OUT/'local_maps.npz'); valid=np.load(OUT/'valid_patch_indices.npy'); plan=json.loads((OUT/'reference_plan.json').read_text(encoding='utf8')); normals=read(R/'A3/结果/train_manifest.csv'); normals=normals[normals['二分类标签']=='0'].reset_index(drop=True); cache=json.loads(Path(r'D:\coin_image_cache\dataset_v1_original\manifest.json').read_text()); V=OUT/'可视化'; V.mkdir(exist_ok=True)
examples=[]
def img(r,side):
 key=['正面路径','背面路径'][side]; return Image.open(cache.get(r[key],r[key])).convert('RGB')
def crop(im,patch):
 x=(int(patch)%24+.5)/24; yy=(int(patch)//24+.5)/24; w,h=im.size; rad=1.5/24; box=(max(0,int((x-rad)*w)),max(0,int((yy-rad)*h)),min(w,int((x+rad)*w)),min(h,int((yy+rad)*h)));return im.crop(box),box
vmax=float(np.quantile(maps['distances'],.99))
for cid in ids:
 i=d.index[d['钱币编号']==cid][0]; r=d.iloc[i]; distances=maps['distances'][i]; matches=maps['matches'][i]; rp=plan['plans'][r.category_name+'|'+r.category_reason]
 fig,axes=plt.subplots(2,6,figsize=(17,7.5)); details=[]
 for side in [0,1]:
  im=img(r,side); patchidx=int(np.argmax(distances[1,side])); patch=int(valid[patchidx]); nn=int(matches[1,side,patchidx]); refidx=rp['ids'][nn//len(valid)]; ref=normals.iloc[refidx]; refpatch=int(valid[nn%len(valid)]); rim=img(ref,side); qc,box=crop(im,patch); rc,rbox=crop(rim,refpatch)
  axes[side,0].imshow(im);axes[side,0].add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,color='red',linewidth=2))
  for method in [0,1]:
   heat=np.full(576,np.nan);heat[valid]=distances[method,side]; ax=axes[side,method+1]; ax.imshow(im); ax.imshow(heat.reshape(24,24),cmap='inferno',alpha=.6,vmin=0,vmax=vmax,extent=(0,im.width,im.height,0))
  axes[side,3].imshow(rim);axes[side,3].add_patch(Rectangle((rbox[0],rbox[1]),rbox[2]-rbox[0],rbox[3]-rbox[1],fill=False,color='blue',linewidth=2));axes[side,4].imshow(qc);axes[side,5].imshow(rc)
  for j,title in enumerate(['待测原图 / '+['正面','背面'][side],'A5a 差异热图','A5b 差异热图','正常参照 '+ref['钱币编号'],'最大差异位置邻域','最近邻参照邻域']):axes[side,j].set_title(title,fontsize=10);axes[side,j].axis('off')
  details.append({'side':side,'reference_id':ref['钱币编号'],'query_patch':patch,'reference_patch':refpatch,'distance':float(distances[1,side,patchidx])})
 title=f"{cid} | 标签 {r['二分类标签']} / {r['grade_group']} / {r['异常代码']} | {r['category_name']}\nA1={scores['A1'][i]:.4f}  A0+A1={scores['A0+A1'][i]:.4f}  A5a={scores['A5a'][i]:.4f}  A5b={scores['A5b'][i]:.4f}  融合={scores['A0+A1+A5b'][i]:.4f}\n参照层级: {rp['prefix']} | 热图统一色标 0—{vmax:.3f}；亮色仅表示特征差异，非专家损伤标注"
 fig.suptitle(title,fontsize=10);fig.subplots_adjust(left=.015,right=.99,bottom=.025,top=.79,wspace=.08,hspace=.23);fig.savefig(V/(cid+'.png'),dpi=130);plt.close(fig);examples.append({'id':cid,'details':details})
save(OUT/'visualization_examples.json',examples)
fig,ax=plt.subplots(figsize=(8,5))
for name in ['A1','A3','A0+A1','A5a','A5b','A0+A1+A5a','A0+A1+A5b']:
 sub=matched[matched.model==name];ax.plot(sub.recall,sub.fp,marker='o',label=name)
ax.set(xlabel='验证异常召回率',ylabel='正常误报枚数',title='固定召回要求下的误报；同一验证集探索');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(V/'matched_recall.png',dpi=150);plt.close(fig)
print(table(metrics[['model','auroc','ap','f1','recall','fp','fn']]))
print('FALLBACK',summary);print('FUSION',chosen)
# Report tables saved separately for final narrative.
(OUT/'tables.md').write_text(table(metrics[['model','auroc','ap','f1','recall','fp','fn']])+'\n\n'+table(matched[matched.point==str(154/199)][['model','recall','fp','fn']])+'\n\n'+table(groups[(groups.point==str(154/199))&groups.group.isin(['AU','XF','92','97','98'])][['model','group','n','errors']]),encoding='utf8')


