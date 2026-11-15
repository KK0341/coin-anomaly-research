from common import *
prefix=next(p for p in PLANS if p!='GLOBAL');im=loadimg(normal.iloc[PLANS[prefix]['anchor']],0)
M=cv2.getRotationMatrix2D((512,512),5,1.02);M[:,2]+=np.array([10,-8]);disturbed=cv2.warpAffine(im,M,(1024,1024),borderValue=(255,255,255));fixed,rec=align(disturbed,prefix,0)
prod=np.vstack([rec['matrix'],[0,0,1]])@np.vstack([M,[0,0,1]]);pts=np.array([[256,256,1],[768,256,1],[256,768,1],[768,768,1]])
error=float(np.linalg.norm(pts@prod[:2].T-pts[:,:2],axis=1).mean());save(OUT/'synthetic_registration_check.json',{'accepted':rec['accepted'],'mean_corner_error_pixels':error,'record':rec});print(rec['accepted'],error)
assert rec['accepted'] and error<5
save(OUT/'pilot_decision.json',{'proceed':True,'basis':'two of eight real front images pass conservative gradient ECC; six explicitly fallback; synthetic known transform error<5px; original crops visibly preserve finer texture; no anomaly metric inspected','limitations':'high real fallback expected; no expert damage localization; fixed protocol after visual pilot'})
