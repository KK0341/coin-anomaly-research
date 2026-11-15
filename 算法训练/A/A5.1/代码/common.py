import json,hashlib,math,time,sys
from pathlib import Path
import numpy as np,pandas as pd,cv2
from PIL import Image
R=Path(r'E:\异常检测\算法训练'); O=R/'A5.1'; OUT=O/'结果'; C=O/'缓存'
SEEDS=[42,123,2026,3407,7777]
cv2.setNumThreads(2);cv2.setRNGSeed(42)
def read(p):return pd.read_csv(p,dtype=str).fillna('')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def data(split):
 a=read(R/f'A3/结果/{split}_manifest.csv');b=read(R/f'A2/结果/{split}_manifest.csv');assert a['钱币编号'].tolist()==b['钱币编号'].tolist()
 for k in ['category_name','category_reason']:a[k]=b[k]
 return a
train=data('train');val=data('val');normal=train[train['二分类标签']=='0'].reset_index(drop=True);cache=json.loads(Path(r'D:\coin_image_cache\dataset_v1_original\manifest.json').read_text())
old=json.loads((R/'A5/结果/reference_plan.json').read_text(encoding='utf8'))
def loadimg(r,side):
 p=r[['正面路径','背面路径'][side]]
 return np.array(Image.open(cache.get(p,p)).convert('RGB'))
def pool(prefix):
 if prefix=='GLOBAL':return np.arange(len(normal))
 return np.flatnonzero((normal.category_reason=='confirmed')&((normal.category_name==prefix)|normal.category_name.str.startswith(prefix+' → ')))
def key(prefix):return hashlib.sha256(prefix.encode()).hexdigest()[:12]
def plans():
 result={}
 for prefix in sorted(set(p['prefix'] for p in old['plans'].values())|{'GLOBAL'}):
  ids=pool(prefix);result[prefix]={'anchor':int(ids[0]),'seeds':{str(s):np.random.default_rng(s).choice(ids,32,replace=False).tolist() for s in SEEDS}}
 return result
PLANS=plans()
def prefix_for(r):return old['plans'][r.category_name+'|'+r.category_reason]['prefix']
sift=cv2.SIFT_create(nfeatures=1000)
def describe(im):
 gray=cv2.cvtColor(cv2.resize(im,(512,512)),cv2.COLOR_RGB2GRAY); yy,xx=np.mgrid[:512,:512];mask=(((xx-256)**2+(yy-256)**2)<(230**2)).astype('uint8')*255
 kp,des=sift.detectAndCompute(gray,mask);return np.array([k.pt for k in kp],dtype=np.float32),des
anchor_cache={}
def align_sift(im,prefix,side):
 ai=PLANS[prefix]['anchor'];aid=normal.iloc[ai]['钱币编号'];ak=(prefix,side)
 if ak not in anchor_cache:anchor_cache[ak]=describe(loadimg(normal.iloc[ai],side))
 dst,dd=anchor_cache[ak];src,sd=describe(im);record={'anchor_id':aid,'accepted':False,'reason':'too_few_matches','matches':0,'inliers':0,'rotation':0.,'scale':1.,'center_shift':0.,'before_error':None,'after_error':None};M=np.eye(2,3,dtype=np.float64)
 if sd is not None and dd is not None and len(dd)>2:
  matches=cv2.BFMatcher().knnMatch(sd,dd,k=2);good=[m for m,n in matches if m.distance<.75*n.distance];record['matches']=len(good)
  if len(good)>=12:
   a=src[[m.queryIdx for m in good]];b=dst[[m.trainIdx for m in good]];cv2.setRNGSeed(42)
   est,inl=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=2000,confidence=.99,refineIters=10)
   if est is not None:
    ok=inl.ravel().astype(bool);count=int(ok.sum());angle=float(np.degrees(np.arctan2(est[1,0],est[0,0])));scale=float(np.hypot(est[0,0],est[1,0]));shift=float(np.linalg.norm(est@np.array([256,256,1])-256)/512);err=float(np.median(np.linalg.norm(a[ok]@est[:,:2].T+est[:,2]-b[ok],axis=1)));before=float(np.median(np.linalg.norm(a[ok]-b[ok],axis=1)));coverage=float(cv2.contourArea(cv2.convexHull(a[ok]))/(512**2)) if count>=3 else 0
    accept=count>=12 and count/len(good)>=.35 and abs(angle)<=20 and .9<=scale<=1.1 and shift<=.08 and err<=2.5 and coverage>=.08
    record.update(accepted=bool(accept),reason='ok' if accept else 'quality_gate',inliers=count,rotation=angle,scale=scale,center_shift=shift,before_error=before,after_error=err,coverage=coverage)
    if accept:M=est
 M=M.copy();M[:,2]*=2;record['matrix']=M.tolist()
 # Dataset dimensions are checked; use one warp only, no nonlinear deformation.
 assert im.shape[:2]==(1024,1024)
 warped=cv2.warpAffine(im,M,(1024,1024),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255)) if record['accepted'] else im.copy()
 return warped,record

def circle(im):
 gray=cv2.cvtColor(im,cv2.COLOR_RGB2GRAY);binary=(gray<245).astype('uint8')*255
 contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cnt=max(contours,key=cv2.contourArea);(x,y),rad=cv2.minEnclosingCircle(cnt);return x,y,rad

def align(im,prefix,side):
 assert im.shape[:2]==(1024,1024)
 ai=PLANS[prefix]['anchor'];aid=normal.iloc[ai]['钱币编号'];ak=('ecc',prefix,side)
 if ak not in anchor_cache:anchor_cache[ak]=loadimg(normal.iloc[ai],side)
 anchor=anchor_cache[ak];sx,sy,sr=circle(im);tx,ty,tr=circle(anchor);scale=tr/sr
 record={'anchor_id':aid,'accepted':False,'reason':'quality_gate','matches':None,'inliers':None,'scale':float(scale),'rotation':0.,'center_shift':None,'before_error':None,'after_error':None}
 M=np.array([[scale,0,tx-scale*sx],[0,scale,ty-scale*sy]],dtype=np.float32)
 def gray(z):
  z=cv2.cvtColor(cv2.resize(z,(256,256)),cv2.COLOR_RGB2GRAY).astype('float32')/255
  z=cv2.GaussianBlur(z,(5,5),1); gx=cv2.Sobel(z,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(z,cv2.CV_32F,0,1,ksize=3); return cv2.GaussianBlur(cv2.magnitude(gx,gy),(5,5),1)
 warped=cv2.warpAffine(im,M,(1024,1024),borderValue=(255,255,255));a=gray(anchor);b=gray(warped);yy,xx=np.mgrid[:256,:256];mask=(((xx-128)**2+(yy-128)**2)<105**2).astype('uint8')*255
 try:
  cc,W=cv2.findTransformECC(a,b,np.eye(2,3,dtype='float32'),cv2.MOTION_EUCLIDEAN,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,100,1e-5),mask,5)
  W=cv2.invertAffineTransform(W);W[:,2]*=4;full=(np.vstack([W,[0,0,1]])@np.vstack([M,[0,0,1]]))[:2];angle=float(np.degrees(np.arctan2(full[1,0],full[0,0])));shift=float(np.linalg.norm(full@np.array([512,512,1])-512)/1024)
  candidate=cv2.warpAffine(im,full,(1024,1024),borderValue=(255,255,255));idx=mask>0;before=float(np.corrcoef(a[idx],gray(im)[idx])[0,1]);after=float(np.corrcoef(a[idx],gray(candidate)[idx])[0,1])
  accepted=.9<=scale<=1.1 and abs(angle)<=20 and shift<=.08 and cc>=.45 and after>=before-.01
  record.update(accepted=bool(accepted),reason='ok' if accepted else 'quality_gate',rotation=angle,center_shift=shift,before_correlation=before,after_correlation=after,ecc=float(cc))
  if accepted:M=full;warped=candidate
 except cv2.error:record['reason']='ecc_failed'
 if not record['accepted']:M=np.eye(2,3);warped=im.copy()
 record['matrix']=M.tolist();return warped,record

if __name__=='__main__':
 save(O/'reference_plans.json',PLANS)
 protocol={'seeds':SEEDS,'K':32,'A5b':'unchanged A5 features and unrestricted same-side patch NN','A5c':'coin silhouette center/scale + ECC Euclidean to fixed first normal training anchor per reference prefix; 384 whole image; same-side NN within +/-1 of 24-grid','A5d':'same alignment; four non-overlapping 512 crops from 1024, each resized384; concatenate 24x24 maps into48x48; NN within +/-2 grid cells','layer':'A1 frozen net.features[5],3x3 mean,L2; no training','registration_gate':'|rotation|<=20deg,scale.9..1.1,center shift<=.08,ECC>=.45,post NCC>=pre NCC-.01; 256px smoothed gradient magnitude,central radius105 mask; no local deformation','fallback':'identity warp on rejected registration; unrestricted same-side search for query-reference pair if either registration rejected','spatial_mask':'same radius .46 on each grid','aggregation':'top5% per side mean,then max sides','fusion_weight':.1,'normalization':'per method+seed validation empirical midrank CDF, (rank-.5)/755; knots saved; no label used for normalization','selection':'no seed/weight/method tuning; every seed reported','extra_control':'A5a unrestricted category diagnostic, same five seeds, to test category advantage','test_access':False}
 save(O/'protocol.json',protocol)
 inputs=[R/'A1/模型/best.pt',R/'A5/protocol.json',R/'A5/实验报告.md',R/'A5/结果/reference_plan.json',R/'A5/结果/local_scores.csv',R/'A1/融合对照/A0_A1集成/val_predictions.csv']+[R/f'{a}/结果/{s}_manifest.csv' for a in ['A2','A3'] for s in ['train','val']]
 save(OUT/'input_hashes.json',{str(p):sha(p) for p in inputs})
 # Small fixed inspection first; no full validation scoring before visual QA.
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams['font.sans-serif']=['Microsoft YaHei']
 inspection=json.loads((R/'A5/结果/inspection_rows.json').read_text(encoding='utf8'));allrows=pd.concat([train,val]).set_index('钱币编号',drop=False);records=[]
 fig,axs=plt.subplots(8,4,figsize=(12,23))
 for j,it in enumerate(inspection):
  r=allrows.loc[it['钱币编号']];prefix=prefix_for(r) if it['split']=='val' else r.category_name
  if prefix not in PLANS:prefix=next((p for p in PLANS if r.category_name.startswith(p)), 'GLOBAL')
  im=loadimg(r,0);warped,rec=align(im,prefix,0);anchor=loadimg(normal.iloc[PLANS[prefix]['anchor']],0);rec.update(id=r['钱币编号'],split=it['split'],prefix=prefix);records.append(rec)
  axs[j,0].imshow(im);axs[j,1].imshow((warped.astype(float)*.5+anchor*.5).astype('uint8'));axs[j,2].imshow(cv2.resize(cv2.resize(warped,(384,384)),(1024,1024))[256:512,384:640]);axs[j,3].imshow(warped[256:512,384:640])
  for k,title in enumerate([r['钱币编号']+' 原图',f"配准叠图 {'通过' if rec['accepted'] else '回退'}",'整图384再放大局部','1024源图局部']):axs[j,k].set_title(title,fontsize=9);axs[j,k].axis('off')
 fig.tight_layout();fig.savefig(OUT/'pilot.png',dpi=110);save(OUT/'pilot_registration.json',records);print([(x['id'],x['accepted'],x['inliers'],x['before_error'],x['after_error']) for x in records],flush=True)


