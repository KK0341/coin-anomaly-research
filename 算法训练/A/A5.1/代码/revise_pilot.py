from pathlib import Path
p=Path(r'E:\异常检测\算法训练\A5.1\代码\common.py');s=p.read_text(encoding='utf-8-sig');s=s.replace('def align(im,prefix,side):','def align_sift(im,prefix,side):')
insert='''
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
  return cv2.GaussianBlur(z,(9,9),2)
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
'''
s=s.replace("if __name__=='__main__':",insert+"\nif __name__=='__main__':")
s=s.replace('SIFT/RANSAC similarity to fixed first normal training anchor per reference prefix','coin silhouette center/scale + ECC Euclidean to fixed first normal training anchor per reference prefix').replace('matches/inliers>=12,inlier fraction>=.35,|rotation|<=20deg,scale.9..1.1,center shift<=.08,median residual<=2.5px at512,source inlier hull>=.08 image','|rotation|<=20deg,scale.9..1.1,center shift<=.08,ECC>=.45,post NCC>=pre NCC-.01; 256px Gaussian-smoothed gray,central radius105 mask; no local deformation')
p.write_text(s,encoding='utf8')
