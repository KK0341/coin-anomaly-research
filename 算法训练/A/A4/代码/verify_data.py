import json,csv,hashlib,py_compile
from pathlib import Path
R=Path(__file__).resolve().parents[1];B=R.parent
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
checks={};ids={}; cache=json.loads(Path(r'D:\coin_image_cache\dataset_v1_original\manifest.json').read_text(encoding='utf-8'))
for split,fn in [('train','训练集.csv'),('val','验证集.csv'),('test','测试集.csv')]:
 original=read(B/'A3/结果'/f'{split}_manifest.csv');new=read(R/'A4a/结果'/f'{split}_manifest.csv');other=read(R/'A4b/结果'/f'{split}_manifest.csv');assert new==other
 assert all(all(r[k]==s[k] for k in r) for r,s in zip(original,new));assert len(original)==len(new)
 ids[split]={r['证书编号'] for r in new};assert len(ids[split])==len(new)
 assert all(r[side] in cache and Path(cache[r[side]]).exists() for r in new for side in ['正面路径','背面路径'])
 c=json.loads((R/'A4a/模型/config.json').read_text(encoding='utf-8'));h=hashlib.sha256((B.parent/'数据划分'/fn).read_bytes()).hexdigest();assert h==c['split_sha256'][split]
 checks[split]={'n':len(new),'original_A3_fields_unchanged':True,'paired_images_cached':True,'split_hash':h}
assert not(ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test'])
for p in R.rglob('*.py'):py_compile.compile(str(p),doraise=True)
checks['certificate_disjoint']=True
(R/'结果/data_verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8');print(checks)
