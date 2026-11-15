import os,json,hashlib,time,math
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
from torchvision.models import convnext_tiny
from torchvision import transforms as T
from PIL import Image
R=Path(r'E:\异常检测\算法训练'); O=R/'A5'; OUT=O/'结果'; CACHE=O/'特征缓存'; CACHE.mkdir(exist_ok=True)
torch.set_num_threads(4); torch.manual_seed(42)
def read(p): return pd.read_csv(p,dtype=str).fillna('')
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def data(split):
 a=read(R/f'A3/结果/{split}_manifest.csv'); b=read(R/f'A2/结果/{split}_manifest.csv')
 assert a['钱币编号'].tolist()==b['钱币编号'].tolist()
 for k in ['category_name','category_reason','原类别']: a[k]=b[k]
 return a
train=data('train'); val=data('val'); normal=train[train['二分类标签']=='0'].reset_index(drop=True)
assert set(train['钱币编号']).isdisjoint(val['钱币编号'])
cache=json.loads(Path(r'D:\coin_image_cache\dataset_v1_original\manifest.json').read_text())
K=32
# Reference sampling uses only normal train membership/category and a fixed seed.
global_ids=np.random.default_rng(42).choice(len(normal),K,replace=False).tolist()
plans={}; refrows=[]
for cat,reason in val[['category_name','category_reason']].drop_duplicates().itertuples(index=False,name=None):
 key=cat+'|'+reason; pool=[]; depth=0; prefix=cat
 if reason=='confirmed':
  parts=cat.split(' → ')
  for d in range(len(parts),2,-1):
   prefix=' → '.join(parts[:d]); pool=np.flatnonzero((normal.category_reason=='confirmed') & ((normal.category_name==prefix)|normal.category_name.str.startswith(prefix+' → '))).tolist()
   if len(pool)>=K: depth=len(parts)-d;break
  else: pool=[]
 if not pool: pool=list(range(len(normal))); prefix='GLOBAL'; depth=-1
 ids=global_ids if prefix=='GLOBAL' else np.random.default_rng(42).choice(pool,K,replace=False).tolist()
 plans[key]={'ids':ids,'prefix':prefix,'fallback_depth':depth,'pool_n':len(pool)}
 for j in ids: refrows.append({'query_category':cat,'category_reason':reason,'prefix':prefix,'fallback_depth':depth,'reference_id':normal.iloc[j]['钱币编号']})
pd.DataFrame(refrows).to_csv(OUT/'reference_membership.csv',index=False,encoding='utf-8-sig')
used=sorted(set(global_ids+sum([p['ids'] for p in plans.values()],[])))
normal.iloc[used].to_csv(OUT/'normal_bank_manifest.csv',index=False,encoding='utf-8-sig'); val.to_csv(OUT/'val_manifest.csv',index=False,encoding='utf-8-sig')
protocol={'backbone':'A1 best.pt frozen','layer':'net.features[5]: 384x24x24; 3x3 stride1 mean then per-patch L2','reference_coins_per_query':K,'reference_seed':42,'mask':'patch centers within radius .46 of image center','distance':'cosine 1-dot nearest patch, same side only, any position','score':'mean top 5% valid patches per side, max of two sides','category':'A2 confirmed true category only; shortest ancestor with >=32 known normal train coins; unknown global','selection':'no layer/K/seed search; fusion weight 0,.1,.25,.5,.75,1 by validation AUROC then AP then smaller weight','test':'NO test images/predictions/manifests read by A5','normal_train_total':len(normal),'normal_bank_union':len(used),'val_n':len(val),'weights_sha256':sha(R/'A1/模型/best.pt')}
save(O/'protocol.json',protocol)
input_paths=[R/'A1/模型/best.pt']+[R/f'{a}/结果/{s}_manifest.csv' for a in ['A2','A3'] for s in ['train','val']]+[R/f'{a}/结果/val_predictions.csv' for a in ['A0','A1','A3']]+[R/'A1/融合对照/A0_A1集成/val_predictions.csv']
save(OUT/'input_hashes.json',{str(p):sha(p) for p in input_paths})
trans=T.Compose([T.Resize((384,384),antialias=True),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
model=convnext_tiny(weights=None); ck=torch.load(R/'A1/模型/best.pt',map_location='cpu',weights_only=False)['model']
state={k[4:]:v for k,v in ck.items() if k.startswith('net.')}; model.classifier[2]=torch.nn.Identity(); model.load_state_dict(state,strict=True); model=model.features[:6].eval().cuda(); del ck,state
rows=pd.concat([normal.iloc[used],val]).reset_index(drop=True)
@torch.inference_mode()
def extract():
 missing=[r for r in rows.to_dict('records') if not (CACHE/(r['钱币编号']+'.npy')).exists()]
 print('EXTRACT',len(missing),'bank',len(used),flush=True); t=time.time()
 for st in range(0,len(missing),8):
  batch=missing[st:st+8]; images=[]
  for r in batch:
   for side in ['正面路径','背面路径']:
    with Image.open(cache.get(r[side],r[side])) as im: images.append(trans(im.convert('RGB')))
  with torch.autocast('cuda',dtype=torch.float16): z=model(torch.stack(images).cuda()); z=F.avg_pool2d(z,3,1,1)
  z=F.normalize(z.float(),dim=1).permute(0,2,3,1).reshape(len(batch),2,576,384).cpu().numpy().astype('float16')
  for r,f in zip(batch,z): np.save(CACHE/(r['钱币编号']+'.npy'),f)
  if st%80==0: print('extract',st+len(batch),'/',len(missing),'sec',round(time.time()-t),flush=True)
extract(); del model; torch.cuda.empty_cache()
coords=np.array([(x,y) for y in range(24) for x in range(24)]); mask=((coords+.5)/24-.5)**2; valid=np.flatnonzero(mask.sum(1)<=.46**2); np.save(OUT/'valid_patch_indices.npy',valid)
@torch.inference_mode()
def score():
 results=[]; maps=[]; matches=[]
 bank_cache={}
 for i,r in enumerate(val.to_dict('records')):
  plan=plans[r['category_name']+'|'+r['category_reason']]; q=torch.tensor(np.load(CACHE/(r['钱币编号']+'.npy'))[:,valid],device='cuda').float(); q=F.normalize(q,dim=-1)
  rec={'钱币编号':r['钱币编号'],'fallback_depth':plan['fallback_depth'],'reference_prefix':plan['prefix'],'pool_n':plan['pool_n']}; both=[]; matched=[]
  for name,ids in [('A5a',global_ids),('A5b',plan['ids'])]:
   key=tuple(ids)
   if key not in bank_cache:
    b=np.stack([np.load(CACHE/(normal.iloc[j]['钱币编号']+'.npy'))[:,valid] for j in ids]); bank_cache[key]=b
   b=torch.tensor(bank_cache[key],device='cuda').float().permute(1,0,2,3).reshape(2,-1,384); b=F.normalize(b,dim=-1)
   sim=q@b.transpose(1,2); best,ix=sim.max(-1); dist=(1-best).clamp_min(0); rec[name]=dist.topk(math.ceil(len(valid)*.05),dim=1).values.mean(1).max().item()
   both.append(dist.cpu().numpy()); matched.append(ix.cpu().numpy())
  results.append(rec); maps.append(both); matches.append(matched)
  if i%100==0: print('score',i,'/',len(val),flush=True)
 pd.DataFrame(results).to_csv(OUT/'local_scores.csv',index=False,encoding='utf-8-sig'); np.savez_compressed(OUT/'local_maps.npz',distances=np.array(maps),matches=np.array(matches)); save(OUT/'reference_plan.json',{'global_ids':global_ids,'plans':plans,'normal_ids':normal['钱币编号'].tolist()})
score(); save(OUT/'extraction_complete.json',{'files':len(rows),'valid_patches_per_side':len(valid),'input_hashes_unchanged':all(sha(Path(p))==v for p,v in json.loads((OUT/'input_hashes.json').read_text()).items())})
print('DONE',flush=True)
