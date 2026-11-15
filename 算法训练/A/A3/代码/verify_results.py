import hashlib,json,csv
import numpy as np
import train as t
from evaluate import load,met
R=t.ROOT;O=t.OUT
hashes=json.loads((O/'source_hashes.json').read_text(encoding='utf-8'))
changed=[p for p,h in hashes.items() if hashlib.sha256(t.Path(p).read_bytes()).hexdigest()!=h];assert not changed,changed
cfg=json.loads((t.MODEL/'config.json').read_text(encoding='utf-8'))
for s,f in t.FILES.items():assert hashlib.sha256((t.SPLIT/f).read_bytes()).hexdigest()==cfg['split_sha256'][s]
h=json.loads((O/'history.json').read_text(encoding='utf-8'));best=t.torch.load(t.MODEL/'best.pt',map_location='cpu',weights_only=False)
assert best['epoch']==max(h,key=lambda r:r['val_auroc'])['epoch']
last=t.torch.load(t.MODEL/'last.pt',map_location='cpu',weights_only=False);assert last['epoch']==len(h)
steps={int(v['step']) for v in last['optimizer']['state'].values()};assert steps=={len(h)*755},steps
state=t.torch.load(R.parent/'A0/模型/convnext_tiny-983f1562.pth',map_location='cpu',weights_only=True)
changed_backbone=sum(not t.torch.equal(v,state[k[4:]]) for k,v in best['model'].items() if k.startswith('net.') and k[4:] in state);assert changed_backbone>0
m=t.Dual().cuda();m.load_state_dict(best['model']);vals=t.predict(m,t.loader(t.Coins('val'),8,workers=0));y,p=load('A3','val');delta=float(np.max(abs(vals[1]-p)));assert delta<1e-7
fr=json.loads((t.MODEL/'frozen_evaluation.json').read_text(encoding='utf-8'));assert (O/'test_predictions.csv').stat().st_mtime>=fr['time_unix']
out={'historical_files_unchanged':len(hashes),'split_hashes_unchanged':True,'epochs':len(h),'optimizer_step':list(steps),'changed_backbone_tensors':changed_backbone,'validation_reload_max_abs_difference':delta,'test_after_frozen_selection':True,'test_not_used_for_selection':True}
t.save_json(O/'verification.json',out);t.save_json(O/'COMPLETE.json',out);print(out)
