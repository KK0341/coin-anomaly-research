import train as t
import torch,json,time
from pathlib import Path
t.seed(); m=t.pretrained(); opt=t.optimizer(m); scaler=torch.amp.GradScaler('cuda',init_scale=1024)
ds=t.Coins('train',True); ds.epoch[0]=1; dl=t.loader(ds,2,workers=0)
x,y,ix=next(iter(dl)); assert x.shape==(2,2,3,504,504)
with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
 f=m.net(x.cuda().flatten(0,1)); assert f.shape==(4,384)
watch=['patch_embed.proj.weight','blocks.0.attn.qkv.weight','blocks.11.mlp.fc2.weight','norm.weight']
before={k:dict(m.net.named_parameters())[k].detach().cpu().clone() for k in watch}; gradients={}
handles=[]
for k in watch:
 def hook(g,k=k):
  if torch.isfinite(g).all(): gradients[k]=float(g.float().norm())
 handles.append(dict(m.net.named_parameters())[k].register_hook(hook))
torch.cuda.reset_peak_memory_stats(); result=t.train_epoch(m,opt,scaler,dl,accum=4,limit=4)
changes={k:float((dict(m.net.named_parameters())[k].detach().cpu()-v).abs().max()) for k,v in before.items()}
assert all(v>0 for v in changes.values()) and all(gradients.get(k,0)>0 for k in watch)
for h in handles: h.remove()
result.update(input_shape=list(x.shape),feature_shape=list(f.shape),grad_norms=gradients,max_parameter_changes=changes,effective_batch=8,micro_batch=2,accum=4,mask_token='requires_grad True but unused in unmasked inference',probe_not_reused_for_training=True)
t.save_json(t.OUT/'smoke.json',result); print(json.dumps(result,indent=2),flush=True)
