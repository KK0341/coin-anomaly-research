import json, hashlib, importlib.util
from pathlib import Path
from PIL import Image
import pandas as pd
R=Path(__file__).resolve().parents[1]
old=json.loads((R.parent/'A1/模型/config.json').read_text(encoding='utf8'))
new=json.loads((R/'模型/config.json').read_text(encoding='utf8'))
diff={k:[old.get(k),new.get(k)] for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
assert set(diff)<= {'resolution','augmentation','training_start_unix_estimate','timing_start_method'}
spec=importlib.util.spec_from_file_location('a8_train',R/'代码/train.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
model=m.Dual(); total=sum(p.numel() for p in model.parameters()); trainable=sum(p.numel() for p in model.parameters() if p.requires_grad); assert total==trainable
splits={k:pd.read_csv(m.SPLIT/v,dtype=str).fillna('') for k,v in m.FILES.items()}
ids={k:set(d['钱币编号']) for k,d in splits.items()}
assert not(ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test'])
cache=json.loads(Path(old['loader']['cache_manifest']).read_text(encoding='utf8')); checked=[]
for k,d in splits.items():
    assert len(d)==len(ids[k])
    for r in d.sample(n=8,random_state=42).to_dict('records'):
        for side in ['正面路径','背面路径']:
            p=Path(r[side]); c=Path(cache[str(p)]); assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(c.read_bytes()).digest()
            with Image.open(c) as im: size=im.size
            checked.append(dict(split=k,id=r['钱币编号'],side=side,size=size,original_cache_identical=True))
result=dict(config_diff=diff,total_parameters=total,trainable_parameters=trainable,split_counts={k:len(d) for k,d in splits.items()},no_coin_overlap=True,cache_sample_checks=checked)
(R/'审计/control_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='cache_sample_checks'},ensure_ascii=False))
