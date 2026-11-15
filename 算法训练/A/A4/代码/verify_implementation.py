import sys,importlib.util,json,torch,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parent
def load(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
out={}
for name,baseline in [('A4a','A1'),('A4b','A3')]:
 t=load(name,R/name/'代码/train.py');b=load(baseline,B/baseline/'代码/train.py')
 t.seed();m=t.Dual();b.seed();ref=b.Dual();s=m.state_dict();assert all(torch.equal(v,s[k]) for k,v in ref.state_dict().items())
 assert (m.grade_head is not None)==(name=='A4b')
 rs=t.rows('train');assert len(rs)==6033
 x=torch.randn(2,1536,requires_grad=True);g=m.reason_head(x);loss=torch.nn.functional.cross_entropy(g,torch.tensor([0,1]));loss.backward();assert x.grad.abs().sum()>0
 out[name]={'baseline_initial_tensors_equal':True,'reason_gradient_to_features':True,'normal_grade_rule':all(int(r['grade_id'])<0 for r in rs if r['二分类标签']=='1'),'train_reason_counts':t.CONFIG['reason_class_weights']}
(R/'结果/implementation_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(out)
