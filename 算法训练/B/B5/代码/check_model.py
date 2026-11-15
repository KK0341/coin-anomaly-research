"""CPU architecture check; cloud smoke remains mandatory."""
import argparse
import json
from pathlib import Path
import torch
import b0_reference as base
from model import Dual

def main():
    p=argparse.ArgumentParser(); p.add_argument('--a8',type=Path,required=True); a=p.parse_args()
    torch.set_num_threads(2)
    base.seed(42); reference=base.Dual()
    base.seed(42); model=Dual()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in reference.state_dict().items())
    model.initialize_a8(a.a8)
    reference.load_state_dict({k:v for k,v in model.state_dict().items() if k in reference.state_dict()})
    added=sum(v.numel() for v in model.parameters())-sum(v.numel() for v in reference.parameters())
    assert added==9222
    model.eval(); reference.eval()
    x=torch.randn(1,2,3,512,512)
    with torch.no_grad():
        full,global_z,attention=model(x,details=True)
        assert torch.allclose(global_z,reference(x),atol=1e-6)
        assert attention.shape==(1,6,2,16,16)
        assert torch.allclose(attention.flatten(2).sum(-1),torch.ones(1,6),atol=1e-6)
    model.train(); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
    before=model.queries.detach().clone()
    loss=torch.nn.functional.cross_entropy(model(x),torch.tensor([0])); assert torch.isfinite(loss)
    loss.backward()
    for name in ['queries','evidence_weight','evidence_bias']:
        grad=getattr(model,name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,name
    torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True); optimizer.step()
    assert not torch.equal(before,model.queries.detach())
    print(json.dumps(dict(status='passed',device='cpu',resolution=512,added_parameters=added,
                    parameters=sum(v.numel() for v in model.parameters()),loss=float(loss.detach()),
                    global_matches_b0=True,attention_normalized=True,query_gradient_and_update=True)))

if __name__=='__main__': main()
