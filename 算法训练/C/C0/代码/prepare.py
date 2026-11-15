import csv,json,re,hashlib,collections
from pathlib import Path
import openpyxl
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parents[1]; PROJECT=R.parents[2]; A3=R.parent.parent/'A/A3/结果'; B=R.parent.parent/'B/B0/数据缓存'
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),'utf8')
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def wc(p,rs):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def main():
    book=openpyxl.load_workbook(PROJECT/'袁世凯采集记录.xlsx',read_only=True,data_only=True);it=book['钱币详细信息'].iter_rows(values_only=True);head=next(it);ex=collections.defaultdict(set)
    for vals in it:
        r=dict(zip(head,vals));ex[str(r.get('证书号','')).removesuffix('.0').zfill(8)].add(str(r.get('评级','')).strip())
    book.close();cache=json.loads(Path('D:/coin_image_cache/dataset_v1_original/manifest.json').read_text('utf8'))
    manifest={};audit={};seen=set(); valid={'G':{4,6},'F':{12,15},'VF':{20,25,30,35},'XF':{40,45},'AU':{50,53,55,58},'MS':set(range(60,71))}
    distribution=rows(A3/'numeric_grade_distribution.csv')
    for split,fn in [('train','训练集.csv'),('val','验证集.csv'),('test','测试集.csv')]:
        path=PROJECT/'数据划分'/fn;raw=rows(path);old={r['钱币编号']:r for r in rows(A3/f'{split}_manifest.csv')};out=[]
        for r in raw:
            cert=r['证书编号'];assert cert not in seen;seen.add(cert)
            assert all(r[k]==old[r['钱币编号']][k] for k in ['证书编号','原始评级','正常数字等级','二分类标签','正面路径','背面路径'])
            normal=r['二分类标签']=='0';g=r['正常数字等级'];m=re.fullmatch(r'(G|F|VF|XF|AU|MS)(\d+)',g)
            reliable=normal and not r['异常代码'] and m is not None and int(m[2]) in valid[m[1]] and ex[cert]=={r['原始评级']} and (lambda o: o is not None and o[1]==m[1] and int(o[2])==int(m[2]))(re.match(r'^(G|F|VF|XF|AU|MS)(\d+)',r['原始评级']))
            assert reliable==(int(old[r['钱币编号']]['grade_id'])>=0), (cert, r['原始评级'], ex[cert], old[r['钱币编号']])
            category='ordinary' if reliable else ('SP/PR' if normal and re.match(r'^(SP|PR)',r['原始评级']) else 'other_normal' if normal else 'abnormal')
            images=[str(Path(cache.get(r[k],r[k]))) for k in ['正面路径','背面路径']];assert all(Path(p).is_file() for p in images)
            out.append({**{k:r[k] for k in ['钱币编号','证书编号','原始评级','正常数字等级','异常代码','二分类标签','正面路径','背面路径']},'plus':'+' in r['原始评级'],'category':category,'base_grade':g if reliable else '', 'images':images})
        actual=collections.Counter(r['正常数字等级'] for r in raw if r['二分类标签']=='0');assert actual==collections.Counter({r['grade']:int(r['n']) for r in distribution if r['split']==split})
        audit[split]={'normal':sum(r['二分类标签']=='0' for r in out),'categories':dict(collections.Counter(r['category'] for r in out)),'split_sha256':hashlib.sha256(path.read_bytes()).hexdigest()};manifest[split]=out
    labels=sorted({r['base_grade'] for rs in manifest.values() for r in rs if r['base_grade']},key=lambda g:int(re.search(r'\d+',g)[0]));mapping={g:i for i,g in enumerate(labels)}
    for rs in manifest.values():
        for r in rs:r['label']=mapping.get(r['base_grade'],-1)
    counts=[dict(grade=g,label=i,**{s:sum(r['base_grade']==g for r in manifest[s]) for s in manifest}) for i,g in enumerate(labels)]
    save(R/'manifest.json',manifest);save(R/'label_mapping.json',mapping);save(R/'data_audit.json',audit);wc(R/'grade_counts.csv',counts)
    for s,rs in manifest.items():wc(R/f'{s}_manifest.csv',[{**r,'images':json.dumps(r['images'],ensure_ascii=False)} for r in rs])
    samples=[]
    for g in labels:
        samples.append(next(r for rs in manifest.values() for r in rs if r['base_grade']==g))
    sheet=Image.new('RGB',(512,len(samples)*270),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(samples):
        for j,p in enumerate(r['images']):
            with Image.open(p) as im:sheet.paste(im.convert('RGB').resize((256,256)),(j*256,i*270))
        draw.text((0,i*270+256),r['base_grade']+' '+r['钱币编号'],fill='black')
    sheet.save(R/'input_audit.jpg')
    text=['# C 标签与任务清单','沿用 Dataset V1 原始划分、证书与正反面配对；逐条核对 A3 manifest、numeric_grade_distribution.csv 和原 Excel 钱币详细信息。','基础等级为实际普通币标签的并集，不预测 plus；SP/PR、异常原因与 Details 不进入数值等级监督。',json.dumps(audit,ensure_ascii=False),'|等级|train|val|test|训练支持|','|---|---:|---:|---:|---|']
    text += [f"|{c['grade']}|{c['train']}|{c['val']}|{c['test']}|{'有' if c['train'] else '无训练支持，仍纳入固定全类别指标'}|" for c in counts]
    text+=['C0：A8 单模型骨干及融合隐藏层迁移；C1：ImageNet 骨干及随机融合隐藏层。两组均新建 K 类输出层。','固定 seed42，验证全标签 Macro-F1 选最佳，平局取最早轮次；测试不参与选模。','任务：B 阶段完成 → GPU 预检 → C0 → C1 → 自动对比报告。原图缓存按 B 同样方式 resize512，每轮重新计算骨干。','A3 四档参考：验证77.34%、测试78.38%，按异常AUROC选模，非具体等级任务。']
    (R/'标签与任务清单.md').write_text('\n\n'.join(text),'utf8')
    print(json.dumps({'labels':labels,'audit':audit},ensure_ascii=False))
if __name__=='__main__':main()
