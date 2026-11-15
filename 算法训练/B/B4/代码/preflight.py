import torch,json
from pathlib import Path
import train as t
import b0_reference as b
def main():
    t.seed(42);m=t.Dual();t.seed(42);base=b.Dual()
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in base.state_dict().items())
    state=torch.load(t.DATA/'a8.pt',map_location='cpu',weights_only=False)['model'];state={k:v for k,v in state.items() if not k.startswith('head.2.')}
    missing=m.load_state_dict(state,strict=False);base.load_state_dict(state,strict=False)
    assert set(missing.missing_keys)=={'head.2.weight','head.2.bias','centers'}
    m.eval();base.eval();x=torch.randn(2,2,3,64,64)
    with torch.no_grad():assert torch.equal(m(x),base(x))
    feat=torch.randn(8,256,requires_grad=True);y=torch.tensor([0,1,2,3,4,5,0,0]);before=m.centers.detach().clone();opt=torch.optim.AdamW([m.centers],lr=1e-4,weight_decay=0)
    loss=m.center_distances(feat,y).mean();(.1*loss).backward();grad=m.centers.grad.norm(dim=1).tolist();assert min(grad)>0 and feat.grad.norm()>0;opt.step();assert (m.centers.detach()-before).norm()>0
    t.save(t.ROOT/'B4/preflight.json',dict(network_initialization_identical_to_B0=True,A8_load_missing=missing.missing_keys,inference_logits_identical=True,center_gradients=grad,center_update_norm=float((m.centers.detach()-before).norm()),loss=float(loss.detach()),extra_training_parameters=m.centers.numel(),extra_inference_parameters=0))
if __name__=='__main__':main()
