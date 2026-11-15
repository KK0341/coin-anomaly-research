import json,pandas as pd
from pathlib import Path
from PIL import Image,ImageDraw
R=Path(r'E:\异常检测\算法训练'); O=R/'A5'; (O/'代码').mkdir(exist_ok=True); (O/'结果').mkdir(exist_ok=True)
cache=json.loads(Path(r'D:\coin_image_cache\dataset_v1_original\manifest.json').read_text())
rows=[]
for split in ['train','val']:
 d=pd.read_csv(R/f'A3/结果/{split}_manifest.csv',dtype=str).fillna(''); p=pd.read_csv(R/'A1/结果/val_predictions.csv',dtype=str) if split=='val' else None
 for name,mask in [('AU',(d['二分类标签']=='0')&(d.grade_group=='AU')),('XF',(d['二分类标签']=='0')&(d.grade_group=='XF')),('92',d['异常代码'].str.contains('92')),('98',d['异常代码'].str.contains('98'))]:
  sub=d[mask]
  if split=='val':
   sub=sub.merge(p[['钱币编号','coin_score']],on='钱币编号'); sub=sub.sort_values('coin_score',key=lambda x:x.astype(float),ascending=name in ['92','98'])
  r=sub.iloc[0].to_dict(); r.update(split=split,group=name); rows.append(r)
canvas=Image.new('RGB',(1200,len(rows)*310),'white'); draw=ImageDraw.Draw(canvas); audit=[]
for i,r in enumerate(rows):
 for s,key in enumerate(['正面路径','背面路径']):
  im=Image.open(cache.get(r[key],r[key])).convert('RGB'); audit.append(dict(id=r['钱币编号'],split=r['split'],side=s,size=im.size))
  thumb=im.copy(); thumb.thumbnail((290,280)); canvas.paste(thumb,(s*300,i*310+25))
  w,h=im.size; crop=im.crop((int(w*.3),int(h*.3),int(w*.7),int(h*.7))); crop.thumbnail((290,280));canvas.paste(crop,(600+s*300,i*310+25))
 draw.text((5,i*310+3),f"{r['split']} {r['钱币编号']} {r['group']} (full front/back; central crops)",fill='black')
canvas.save(O/'结果/feasibility.jpg'); (O/'结果/feasibility.json').write_text(json.dumps(audit,indent=2)); (O/'结果/inspection_rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
print(audit)
