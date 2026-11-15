from common import *
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams['font.sans-serif']=['Microsoft YaHei'];plt.rcParams['axes.unicode_minus']=False
V=OUT/'可视化';V.mkdir(exist_ok=True)
pr=pd.read_csv(OUT/'predictions.csv',dtype={'钱币编号':str});reg=pd.read_csv(OUT/'registration_audit.csv',dtype={'钱币编号':str});meta=json.loads((OUT/'thresholds.json').read_text());target=str(154/199)
ids=['51062170','38540371','89712429','46742043']
validids=set(val['钱币编号']);cand=reg[reg['钱币编号'].isin(validids)&reg.accepted].groupby('钱币编号').size()
for cid in cand[cand==2].index:
 if cid not in ids:ids.append(cid);break
# One deterioration at matched recall, keep failure evidence.
bp=pr[(pr.model=='融合_A5b')&(pr.seed==42)].set_index('钱币编号');dp=pr[(pr.model=='融合_A5d')&(pr.seed==42)].set_index('钱币编号')
for _,r in val.iterrows():
 cid=r['钱币编号'];y=int(r['二分类标签']);b=bp.loc[cid].score>=meta['融合_A5b_42'][target];d=dp.loc[cid].score>=meta['融合_A5d_42'][target]
 if b==y and d!=y and cid not in ids:ids.append(cid);break
allrows=val.set_index('钱币编号',drop=False)
def warped(im,rec):return cv2.warpAffine(im,np.array(rec['matrix']),(1024,1024),borderValue=(255,255,255)) if rec['accepted'] else im

def box(patch,n=48):
 x=(int(patch)%n+.5)/n*1024;y=(int(patch)//n+.5)/n*1024;rad=1.5/n*1024;return [int(max(0,x-rad)),int(max(0,y-rad)),int(min(1024,x+rad)),int(min(1024,y+rad))]
examples=[]
for cid in ids:
 r=allrows.loc[cid];prefix=prefix_for(r);folder=C/key(prefix);qr=json.loads((folder/(cid+'.json')).read_text());zc=np.load(OUT/'maps_seed42'/('A5c_'+cid+'.npz'));zd=np.load(OUT/'maps_seed42'/('A5d_'+cid+'.npz'));fig,ax=plt.subplots(2,7,figsize=(21,8));details=[]
 for side in [0,1]:
  p=int(np.argmax(zd['distances'][side]));qpatch=zd['valid'][p];refidx=int(zd['ref_indices'][side,p]);ref=normal.iloc[refidx];rr=json.loads((folder/(ref['钱币编号']+'.json')).read_text());im=loadimg(r,side);rim=loadimg(ref,side);q=warped(im,qr[side]);refim=warped(rim,rr[side]);qb=box(qpatch);rb=box(zd['ref_patches'][side,p]);cropq=q[qb[1]:qb[3],qb[0]:qb[2]];cropr=refim[rb[1]:rb[3],rb[0]:rb[2]]
  ax[side,0].imshow(im);ax[side,1].imshow(rim);ax[side,2].imshow((im.astype(float)*.5+rim*.5).astype('uint8'));ax[side,3].imshow((q.astype(float)*.5+refim*.5).astype('uint8'))
  for j,(z,n) in enumerate([(zc,24),(zd,48)]):
   heat=np.full(n*n,np.nan);heat[z['valid']]=z['distances'][side];ax[side,4+j].imshow(q);ax[side,4+j].imshow(heat.reshape(n,n),extent=[0,1024,1024,0],cmap='inferno',vmin=0,vmax=float(np.quantile(z['distances'],.99)),alpha=.6)
  ax[side,5].add_patch(Rectangle((qb[0],qb[1]),qb[2]-qb[0],qb[3]-qb[1],fill=False,color='cyan',linewidth=2));ax[side,6].imshow(np.concatenate([cv2.resize(cropq,(192,192)),cv2.resize(cropr,(192,192))],axis=1))
  status='局部约束' if qr[side]['accepted'] and rr[side]['accepted'] else '宽松回退'
  titles=['待测 '+['正面','背面'][side],'正常 '+ref['钱币编号'],'配准前叠图','配准后叠图 / '+status,'A5c 差异图','A5d 差异图','待测 / 最近邻参照局部']
  for j,t in enumerate(titles):ax[side,j].set_title(t,fontsize=10);ax[side,j].axis('off')
  details.append({'side':side,'reference_id':ref['钱币编号'],'query_registered':qr[side]['accepted'],'reference_registered':rr[side]['accepted'],'query_patch':int(qpatch),'reference_patch':int(zd['ref_patches'][side,p]),'matching':status})
 ss=pr[(pr['钱币编号']==cid)&(pr.seed==42)&pr.model.isin(['A5b','A5c','A5d','融合_A5b','融合_A5c','融合_A5d'])];st='  '.join(f'{x.model}={x.score:.4f}' for x in ss.itertuples())
 fig.suptitle(f"{cid} | 标签{r['二分类标签']} / {r['grade_group']} / {r['异常代码']} | seed42固定展示，非最优种子\n{prefix}\n{st}\n热图为各图距离99%分位色标，仅比较位置；不代表专家确认损伤。裁剪框不代表网络完整感受野。",fontsize=10);fig.subplots_adjust(top=.77,bottom=.03,left=.01,right=.99,wspace=.07,hspace=.18);fig.savefig(V/(cid+'.png'),dpi=110);plt.close(fig);examples.append({'id':cid,'details':details})
save(OUT/'visualization_examples.json',examples)
sumdf=pd.read_csv(OUT/'summary.csv').set_index('model');fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for j,metric in enumerate(['auroc','ap']):
 names=['A5a','A5b','A5c','A5d'];xx=np.arange(4)
 for shift,prefix,label in [(-.12,'','局部'),(.12,'融合_','10%融合')]:
  sub=sumdf.loc[[prefix+n for n in names]];axes[j].errorbar(xx+shift,sub[metric+'_mean'],yerr=sub[metric+'_std'],fmt='o',capsize=4,label=label)
 axes[j].axhline(sumdf.loc['A0+A1',metric+'_mean'],ls='--',color='gray',label='A0+A1');axes[j].set(xticks=xx,xticklabels=names,ylabel=metric.upper(),title='五种子均值±样本标准差');axes[j].legend()
fig.tight_layout();fig.savefig(V/'seed_stability.png',dpi=150);plt.close(fig)
print('VISUALS_DONE',len(ids))
