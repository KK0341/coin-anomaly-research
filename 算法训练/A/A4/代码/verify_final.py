import sys,json,hashlib,importlib.util,csv,time
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1];B=R.parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8388608),b''):h.update(block)
 return h.hexdigest()
def main():
 out={};old=json.loads((R/'结果/historical_hashes.json').read_text(encoding='utf-8'));bad=[p for p,h in old.items() if sha(Path(p))!=h];assert not bad;out['historical_files_unchanged']=len(old)
 frozen=json.loads((R/'frozen_evaluation.json').read_text(encoding='utf-8'))
 for n in ['A4a','A4b']:
  d=R/n;sp=importlib.util.spec_from_file_location(n,d/'代码/train.py');t=importlib.util.module_from_spec(sp);sys.modules[n]=t;sp.loader.exec_module(t)
  ck=t.torch.load(d/'模型/last.pt',map_location='cpu',weights_only=False);h=json.loads((d/'结果/history.json').read_text(encoding='utf-8'));steps=sorted({int(v['step']) for v in ck['optimizer']['state'].values() if 'step' in v});assert steps==[len(h)*755]
  best=t.torch.load(d/'模型/best.pt',map_location='cpu',weights_only=False);assert best['val_auroc']==max(x['val_auroc'] for x in h);assert sha(d/'模型/best.pt')==frozen['models'][n]['checkpoint_sha256'];m=t.Dual().cuda();m.load_state_dict(best['model']);ds=t.Coins('val');v=t.predict(m,t.loader(ds,8,workers=0))
  with (d/'结果/val_predictions.csv').open(encoding='utf-8-sig',newline='') as f:rs=list(csv.DictReader(f))
  err=float(np.max(np.abs(v[1]-np.array([float(r['coin_score']) for r in rs]))));assert err<1e-6
  start=json.loads((d/'结果/TEST_INFERENCE_STARTED.json').read_text(encoding='utf-8'));assert start['time_unix']>=frozen['created_unix']
  out[n]={'epochs':len(h),'best_epoch':best['epoch'],'optimizer_steps':steps,'validation_reload_max_abs_error':err,'test_after_freeze':True,'checkpoint_hash':sha(d/'模型/best.pt')}
  if n=='A4a':t.save_json(d/'结果/grade_validation.json',{'enabled':False,'note':'A4a has no grade head; inherited placeholder logits are not predictions and must not be interpreted'})
  del ck,best,m;t.torch.cuda.empty_cache()
 (R/'结果/final_verification.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(out)
if __name__=='__main__':main()
