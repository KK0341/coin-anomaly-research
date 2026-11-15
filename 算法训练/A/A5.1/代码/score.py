from common import *
import torch
from torch.nn import functional as F
torch.set_num_threads(4)
PLANS=json.loads((O/'reference_plans.json').read_text(encoding='utf8'));MAP=OUT/'maps_seed42';MAP.mkdir(exist_ok=True);PART=OUT/'score_parts';PART.mkdir(exist_ok=True)

def raw(cid):
 p=R/'A5/特征缓存'/(cid+'.npy')
 if not p.exists():p=C/'raw'/(cid+'.npy')
 return np.load(p)
def get(cid,prefix,method):
 if method in ['A5a','A5b']:return raw(cid),[False,False]
 folder=C/key(prefix);meta=json.loads((folder/(cid+'.json')).read_text(encoding='utf8'))
 with np.load(folder/(cid+'.npz')) as f:z=f['high' if method=='A5d' else 'low']
 return z,[x['accepted'] for x in meta]
def indices(n):
 y,x=np.mgrid[:n,:n];pos=np.stack([x.ravel(),y.ravel()],1);valid=np.flatnonzero((((pos+.5)/n-.5)**2).sum(1)<=.46**2);return valid,pos[valid]
@torch.inference_mode()
def main():
 start=time.time()
 for method in ['A5a','A5b','A5c','A5d']:
  n=48 if method=='A5d' else 24;valid,pos=indices(n);P=len(valid);allowed=torch.tensor((np.abs(pos[:,None]-pos[None,:]).max(-1)<=(2 if n==48 else 1)),device='cuda')
  prefixes=['GLOBAL'] if method=='A5a' else sorted(set(val.apply(prefix_for,axis=1)))
  for pi,prefix in enumerate(prefixes):
   dest=PART/(method+'_'+key(prefix)+'.csv')
   if dest.exists():continue
   plan=PLANS[prefix];ids=plan['union'];refs=[];accept=[]
   if method in ['A5c','A5d']:
    needed=list(normal.iloc[ids]['钱币编号'])+list(val[val.apply(prefix_for,axis=1)==prefix]['钱币编号'])
    while not all((C/key(prefix)/(cid+'.json')).exists() for cid in needed):
     print('WAIT_FEATURES',method,prefix,flush=True);time.sleep(30)
   for j in ids:
    z,ok=get(normal.iloc[j]['钱币编号'],prefix,method);refs.append(z[:,valid]);accept.append(ok)
   bank=F.normalize(torch.tensor(np.stack(refs),device='cuda').float(),dim=-1);accept=np.array(accept);del refs
   subset=val if method=='A5a' else val[val.apply(prefix_for,axis=1)==prefix];records=[]
   for qi,(_,r) in enumerate(subset.iterrows()):
    cid=r['钱币编号'];z,qok=get(cid,prefix,method);q=F.normalize(torch.tensor(z[:,valid],device='cuda').float(),dim=-1);bestall=[];locall=[]
    for side in [0,1]:
     bests=[];locs=[]
     for st in range(0,len(ids),8):
      b=bank[st:st+8,side];sim=torch.matmul(q[side],b.transpose(1,2))
      if method in ['A5c','A5d']:
       good=torch.tensor(accept[st:st+8,side]&qok[side],device='cuda');sim.masked_fill_(good[:,None,None]&(~allowed[None]),-torch.inf)
      best,loc=sim.max(-1);bests.append((1-best).clamp_min(0));locs.append(loc)
     bestall.append(torch.cat(bests));locall.append(torch.cat(locs))
    distances=torch.stack(bestall);locations=torch.stack(locall)
    for seed in SEEDS:
     ix=[ids.index(j) for j in plan['seeds'][str(seed)]];ds,ri=distances[:,ix].min(dim=1);score=ds.topk(math.ceil(P*.05),dim=1).values.mean(1).max().item();fallback=float(np.mean(~(accept[ix]&np.array(qok)[None]))) if method in ['A5c','A5d'] else None
     records.append({'钱币编号':cid,'method':method,'seed':seed,'score':score,'prefix':prefix,'query_front_aligned':qok[0],'query_back_aligned':qok[1],'pair_side_fallback_fraction':fallback})
     if seed==42:
      ri_np=ri.cpu().numpy();matchref=np.array(plan['seeds'][str(seed)])[ri_np];sourceix=np.array(ix)[ri_np];loc_np=locations.cpu().numpy();matchloc=np.stack([loc_np[s,sourceix[s],np.arange(P)] for s in [0,1]])
      np.savez_compressed(MAP/(method+'_'+cid+'.npz'),distances=ds.cpu().numpy(),ref_indices=matchref,ref_patches=valid[matchloc],valid=valid)
    if qi%100==0:print(method,pi+1,'/',len(prefixes),qi,'/',len(subset),'sec',round(time.time()-start),flush=True)
   pd.DataFrame(records).to_csv(dest,index=False,encoding='utf-8-sig');del bank;torch.cuda.empty_cache()
 parts=[pd.read_csv(p,dtype={'钱币编号':str}) for p in PART.glob('*.csv')];all_scores=pd.concat(parts);assert len(all_scores)==755*5*4;all_scores.to_csv(OUT/'local_scores.csv',index=False,encoding='utf-8-sig');save(OUT/'scoring_complete.json',{'seconds':time.time()-start,'rows':len(all_scores)});print('SCORES_DONE',flush=True)
if __name__=='__main__':main()


