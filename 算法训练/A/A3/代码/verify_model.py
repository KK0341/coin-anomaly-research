import importlib.util,sys,json,hashlib
from pathlib import Path
import torch
import train as t
spec=importlib.util.spec_from_file_location('a1',t.ROOT.parent/'A1/代码/train.py');a1=importlib.util.module_from_spec(spec);spec.loader.exec_module(a1)
t.seed();base=a1.pretrained();t.seed();m=t.pretrained()
assert all(torch.equal(v,m.state_dict()[k]) for k,v in base.state_dict().items())
del base
ds=t.Coins('train');ids=[next(i for i,r in enumerate(ds.rows) if int(r['grade_id'])==g) for g in range(4)]
x=torch.stack([ds[i][0] for i in ids]).cuda();m.eval();z,g=m(x)
param=next(m.net.parameters());la=torch.nn.functional.binary_cross_entropy_with_logits(z,torch.zeros_like(z));lg=torch.nn.functional.cross_entropy(g,torch.arange(4,device='cuda'))
ga=torch.autograd.grad(la,param,retain_graph=True)[0];gg=torch.autograd.grad(lg,param)[0]
assert ga.isfinite().all() and gg.isfinite().all() and ga.norm()>0 and gg.norm()>0
out={'A1_initialization_exact_equal':True,'anomaly_backbone_gradient_norm':float(ga.norm()),'grade_backbone_gradient_norm':float(gg.norm()),'forward_inputs':'images only','grade_ids_not_forward_inputs':True}
t.save_json(t.OUT/'model_verification.json',out);print(out)
