from common import *
import torch
from torch.nn import functional as F
from torchvision.models import convnext_tiny
from torchvision import transforms as T
torch.set_num_threads(4);torch.manual_seed(42)
trans=T.Compose([T.Resize((384,384),antialias=True),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
model=convnext_tiny(weights=None);model.classifier[2]=torch.nn.Identity();state=torch.load(R/'A1/模型/best.pt',map_location='cpu',weights_only=False)['model'];model.load_state_dict({k[4:]:v for k,v in state.items() if k.startswith('net.')});model=model.features[:6].eval().cuda();del state
@torch.inference_mode()
def features(images):
 out=[]
 for st in range(0,len(images),12):
  x=torch.stack([trans(Image.fromarray(im)) for im in images[st:st+12]]).cuda()
  with torch.autocast('cuda',dtype=torch.float16):z=F.avg_pool2d(model(x),3,1,1)
  out.append(F.normalize(z.float(),dim=1).permute(0,2,3,1).cpu().numpy().astype('float16'))
 return np.concatenate(out)
rows={r['钱币编号']:r for r in pd.concat([normal,val]).to_dict('records')}
raw_ids=set(val['钱币编号']);tasks={}
for prefix,p in PLANS.items():
 ids=sorted(set(sum(p['seeds'].values(),[])));p['union']=ids
 qids=val[val.apply(prefix_for,axis=1)==prefix]['钱币编号'].tolist()
 coins=set(normal.iloc[ids]['钱币编号'])|set(qids);raw_ids|=coins
 # GLOBAL also serves A5a; aligned features only needed for actual global-fallback queries.
 if not qids:continue
 tasks[prefix]=sorted(coins)
save(O/'reference_plans.json',PLANS);save(OUT/'extraction_tasks.json',tasks)
rawdir=C/'raw';rawdir.mkdir(exist_ok=True)
print('RAW',len(raw_ids),'ALIGNED',sum(map(len,tasks.values())),flush=True);start=time.time()
for idx,cid in enumerate(sorted(raw_ids)):
 dest=rawdir/(cid+'.npy');oldpath=R/'A5/特征缓存'/(cid+'.npy')
 if dest.exists() or oldpath.exists():continue
 z=features([loadimg(rows[cid],s) for s in [0,1]]);np.save(dest,z.reshape(2,576,384))
 if idx%100==0:print('raw',idx,'sec',round(time.time()-start),flush=True)
for pi,(prefix,cids) in enumerate(tasks.items()):
 folder=C/key(prefix);folder.mkdir(exist_ok=True)
 for i,cid in enumerate(cids):
  dest=folder/(cid+'.npz');meta=folder/(cid+'.json')
  if dest.exists() and meta.exists():continue
  warped=[];records=[]
  for side in [0,1]:
   im,rec=align(loadimg(rows[cid],side),prefix,side);warped.append(im);records.append(rec)
  images=warped+[im[y:y+512,x:x+512] for im in warped for y in [0,512] for x in [0,512]];z=features(images)
  low=z[:2].reshape(2,576,384);tiles=z[2:].reshape(2,2,2,24,24,384);high=tiles.transpose(0,1,3,2,4,5).reshape(2,2304,384)
  np.savez(dest,low=low,high=high);save(meta,records)
  if i%50==0:print('aligned',pi+1,'/',len(tasks),i,'/',len(cids),'sec',round(time.time()-start),flush=True)
save(OUT/'extraction_complete.json',{'seconds':time.time()-start,'raw_coins':len(raw_ids),'alignment_coin_prefix_pairs':sum(map(len,tasks.values()))});print('EXTRACTION_DONE',flush=True)
