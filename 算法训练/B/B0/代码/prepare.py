import csv,json,hashlib,shutil,tarfile
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[2]; P=ROOT/'B0/数据缓存'; P.mkdir(exist_ok=True); (P/'images').mkdir(exist_ok=True)
A=ROOT.parent/'A'; codes=['92','97','98','95','94','91']; counts={'train':[891,267,207,81,75,62],'val':[112,33,26,10,9,8],'test':[112,33,26,10,9,8]}
cache=json.loads(Path('D:/coin_image_cache/dataset_v1_original/manifest.json').read_text('utf8'))
manifest={}; audit={}; ids=set(); hashes=set(); tasks=[]
for split,name in [('train','训练集'),('val','验证集'),('test','测试集')]:
    path=ROOT.parents[1]/'数据划分'/f'{name}.csv'
    with path.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
    localids={r['钱币编号'] for r in rows}; assert len(localids)==len(rows) and not ids&localids; ids|=localids
    localhash={r[k] for r in rows for k in ['正面像素SHA256','背面像素SHA256'] if r[k]}; assert not hashes&localhash; hashes|=localhash
    selected=[r for r in rows if r['二分类标签']=='1' and r['异常代码'] in codes]
    c=Counter(r['异常代码'] for r in selected); assert [c[x] for x in codes]==counts[split],c
    audit[split]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'six_counts':dict(c),'excluded':dict(Counter(r['异常代码'] for r in rows if r['二分类标签']=='1' and r['异常代码'] not in codes))}
    keep=[r for r in rows if r['二分类标签']=='1'] if split=='train' else rows
    manifest[split]=[]
    for r in keep:
        out={k:r[k] for k in ['钱币编号','异常代码','二分类标签','正面路径','背面路径']}
        out['images']=[]
        for side,k in enumerate(['正面路径','背面路径']):
            dest=f"images/{r['钱币编号']}_{side}.png"; out['images'].append(dest); tasks.append((cache.get(r[k],r[k]),P/dest))
        manifest[split].append(out)
    if split!='train': shutil.copy2(A/'A8_高分辨率实验'/'结果'/f'{split}_predictions.csv',P/f'a8_{split}.csv')
def convert(t):
    src,dst=t
    with Image.open(src) as im: im.convert('RGB').resize((512,512),Image.Resampling.BILINEAR).save(dst)
with ThreadPoolExecutor(8) as pool: list(pool.map(convert,tasks))
for src,dst in [(A/'A8_高分辨率实验/模型/best.pt','a8.pt'),(A/'A0/模型/convnext_tiny-983f1562.pth','imagenet.pth')]:
    shutil.copy2(src,P/dst); audit[dst]=hashlib.sha256(src.read_bytes()).hexdigest()
assert audit['imagenet.pth'].startswith('983f1562')
shutil.copy2(A/'A8_高分辨率实验/代码/train.py',P/'a8_reference.py')
shutil.copy2(A/'A8_高分辨率实验/模型/threshold.json',P/'a8_threshold.json')
(P/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False),'utf8')
(P/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),'utf8')
sheet=Image.new('RGB',(1024,6*270),'white'); draw=ImageDraw.Draw(sheet)
for j,c in enumerate(codes):
    rs=[r for r in manifest['train'] if r['异常代码']==c]
    for k,r in enumerate([rs[0],rs[len(rs)//2]]):
        for side,p in enumerate(r['images']):
            with Image.open(P/p) as im: sheet.paste(im.resize((256,256)),(k*512+side*256,j*270))
        draw.text((k*512,j*270+256),c+' '+r['钱币编号'],fill='black')
sheet.save(ROOT/'B0/运行记录/input_preview.jpg')
print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
