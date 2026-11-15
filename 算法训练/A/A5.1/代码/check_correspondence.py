from common import *
import torch
from torch.nn import functional as F
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
# Use first available reliable validation/reference pair, no anomaly score or label selection.
plan=json.loads((O/'reference_plans.json').read_text(encoding='utf8'));chosen=None
for _,r in val.iterrows():
 prefix=prefix_for(r);folder=C/key(prefix);p=folder/(r['钱币编号']+'.json')
 if not p.exists():continue
 qr=json.loads(p.read_text())
 if not qr[0]['accepted']:continue
 for rid in plan[prefix]['seeds']['42']:
  ref=normal.iloc[rid];rp=folder/(ref['钱币编号']+'.json')
  if rp.exists():
   rr=json.loads(rp.read_text())
   if rr[0]['accepted']:chosen=(r,ref,prefix,qr,rr);break
 if chosen:break
assert chosen is not None
r,ref,prefix,qr,rr=chosen;folder=C/key(prefix)
a=np.load(folder/(r['钱币编号']+'.npz'))['low'][0];b=np.load(folder/(ref['钱币编号']+'.npz'))['low'][0];a=F.normalize(torch.tensor(a).float(),dim=-1);b=F.normalize(torch.tensor(b).float(),dim=-1)
positions=[(8,8),(8,15),(15,8),(15,15)];pairs=[]
for yy,xx in positions:
 q=yy*24+xx;allowed=[y*24+x for y in range(yy-1,yy+2) for x in range(xx-1,xx+2)];best=allowed[int((a[q]@b[allowed].T).argmax())];pairs.append((q,best))
ims=[]
for row,meta in [(r,qr),(ref,rr)]:
 im=loadimg(row,0);ims.append(cv2.warpAffine(im,np.array(meta[0]['matrix']),(1024,1024),borderValue=(255,255,255)))
fig,ax=plt.subplots(1,2,figsize=(10,5))
for j,im in enumerate(ims):
 ax[j].imshow(im)
 for color,pair in zip(['red','cyan','lime','yellow'],pairs):
  pt=pair[j];x=(pt%24+.5)*1024/24;y=(pt//24+.5)*1024/24;ax[j].plot(x,y,'o',color=color,markersize=8)
 ax[j].set_title(['配准待测 '+r['钱币编号'],'配准正常参照 '+ref['钱币编号']][j]);ax[j].axis('off')
fig.suptitle('预设四个网格位置：同色点为局部最近邻，仅允许±1格；不判断异常');fig.tight_layout();fig.savefig(OUT/'pilot_correspondence.png',dpi=120)
save(OUT/'pilot_correspondence.json',{'query':r['钱币编号'],'reference':ref['钱币编号'],'prefix':prefix,'grid_pairs':pairs,'selection':'first available accepted validation/reference pair; fixed four positions; no anomaly score selection'});print('CORRESPONDENCE_OK',r['钱币编号'],ref['钱币编号'])
