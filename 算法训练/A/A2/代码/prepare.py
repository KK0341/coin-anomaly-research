from pathlib import Path
import csv,json,hashlib,collections
R=Path(r'E:\异常检测\算法训练\A2'); A1=R.parent/'A1'; S=Path(r'E:\异常检测\数据划分')
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def write(p,rs):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def category(r):
 if r['本次修正组']=='原96枚历史异常' or any(x in (r['待复核依据']+' '+r['原排除原因']) for x in ['分类关联待核对','等待人工确认']):return 'Unknown','historically_unconfirmed'
 if r['分类详情核对'] not in ['一致','匹配来源多编号之一','E前缀差异']:return 'Unknown','unconfirmed'
 if not r['原类别'].strip():return 'Unknown','missing'
 return r['原类别'].strip(),'confirmed'
raw={k:list(csv.DictReader((S/v).open(encoding='utf-8-sig',newline=''))) for k,v in {'train':'训练集.csv','val':'验证集.csv','test':'测试集.csv'}.items()}
names=sorted({category(r)[0] for r in raw['train']}-{'Unknown'}); mapping={'Unknown':0,**{v:i+1 for i,v in enumerate(names)}}
save(R/'模型/category_mapping.json',mapping)
stats=[];audit={}
for split,rs in raw.items():
 out=[]
 for r in rs:
  name,reason=category(r); idx=mapping.get(name,0)
  if name!='Unknown' and idx==0:reason='unseen_in_train'
  out.append({**{k:r[k] for k in ['钱币编号','正面路径','背面路径','二分类标签','原始评级','异常代码','正常数字等级','原类别','版别']},'category_name':name if idx else 'Unknown','category_id':idx,'category_reason':reason})
 write(R/'结果'/f'{split}_manifest.csv',out)
 audit[split]={'total':len(out),'unknown':sum(r['category_id']==0 for r in out),'reasons':dict(collections.Counter(r['category_reason'] for r in out))}
 for name in mapping:
  rr=[r for r in out if r['category_name']==name];n=sum(int(r['二分类标签'])==0 for r in rr);a=len(rr)-n
  stats.append(dict(split=split,category=name,normal=n,abnormal=a,total=len(rr),abnormal_fraction=a/len(rr) if rr else None))
write(R/'结果/category_distribution.csv',stats)
save(R/'结果/category_audit.json',{'known_categories':len(names),'embedding_rows':len(mapping),'splits':audit,'source':'原类别完整路径：年份/面额/细分版别；不使用评级、异常代码、证书号；历史96枚强制Unknown','train_original_categories':len({r['原类别'] for r in raw['train']}),'train_pcgs_varieties':len({r['PCGS品种编号'] for r in raw['train']}),'train_category_depths':dict(collections.Counter(len(r['原类别'].split('→')) for r in raw['train']))})
cfg=json.loads((A1/'模型/config.json').read_text(encoding='utf-8'))
for k in ['training_start_unix_estimate','timing_start_method']:cfg.pop(k,None)
cfg.update(architecture='Shared ConvNeXt-Tiny 768x2 + category Embedding32 -> Linear1568x256 GELU Linear256x1',initialization='A0 original ImageNet convnext_tiny-983f1562.pth; never A1 best.pt',head_parameters='MLP and trainable Embedding, including Unknown',category_count=len(mapping),embedding_dim=32)
save(R/'模型/config.json',cfg)
(R/'结果/loader_selection.json').write_bytes((A1/'结果/loader_selection.json').read_bytes())
source=(A1/'代码/train.py').read_text(encoding='utf-8')
start=source.index('def rows(split):');end=source.index('\nclass Coins',start)
source=source[:start]+'''def rows(split):
    with open(OUT/f'{split}_manifest.csv',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))
'''+source[end:]
source=source.replace("float(r['二分类标签']),i","float(r['二分类标签']),int(r['category_id'])")
source=source.replace('self.head=nn.Sequential(nn.Linear(1536,256)',"self.embedding=nn.Embedding(len(json.loads((MODEL/'category_mapping.json').read_text(encoding='utf-8'))),32)\n        self.head=nn.Sequential(nn.Linear(1568,256)")
source=source.replace('def forward(self,x):','def forward(self,x,category):').replace('z=self.head(features).squeeze(1)','z=self.head(torch.cat([features,self.embedding(category)],dim=1)).squeeze(1)')
source=source.replace("'params':m.head.parameters()","'params':list(m.head.parameters())+list(m.embedding.parameters())")
source=source.replace('for x_cpu,y_cpu,_ in batches:','for x_cpu,y_cpu,c_cpu in batches:').replace('z,_=m(x);','z,_=m(x,c_cpu.cuda(non_blocking=True));')
source=source.replace('for x,y,_ in dl:','for x,y,c in dl:').replace('z,s=m(x.cuda(non_blocking=True))','z,s=m(x.cuda(non_blocking=True),c.cuda(non_blocking=True))')
source=source.replace("# Persist raw test predictions immediately", "save_json(MODEL/'frozen_evaluation.json',{'best_epoch':best['epoch'],'threshold':threshold,'checkpoint_sha256':hashlib.sha256((MODEL/'best.pt').read_bytes()).hexdigest(),'mapping_sha256':hashlib.sha256((MODEL/'category_mapping.json').read_bytes()).hexdigest(),'selection':'validation AUROC; max validation F1 threshold; no test tuning'})\n    # Persist raw test predictions immediately")
(R/'代码/train.py').write_text(source,encoding='utf-8')
# Full historical output integrity baseline (read only).
files=[p for base in [R.parent/'A0',A1] for p in base.rglob('*') if p.is_file()]
save(R/'结果/historical_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
print(json.dumps(json.loads((R/'结果/category_audit.json').read_text(encoding='utf-8')),ensure_ascii=False))
