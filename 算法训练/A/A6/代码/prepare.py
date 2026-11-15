"""A6-1: bounded validation review; no model training or historical test access.
Run with Python + numpy + Pillow. All writes stay in A6; images are copied byte-for-byte.
"""
from pathlib import Path
import csv, json, hashlib, shutil, collections, functools
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT.parent
ID = '钱币编号'
SIDES = ['正面路径', '背面路径']
INPUTS = {}

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(rel):
    p = EXP / rel
    assert 'test' not in str(p).lower() and '测试' not in str(p)
    INPUTS[str(p)] = sha(p)
    with p.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def save(p, obj): p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')
def writecsv(p, rows):
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def key(r): return hashlib.sha256(('A6-1-fixed-order|' + r[ID]).encode()).hexdigest()

def choose(pool, n, anchors=()):
    """Score-quantile coverage + category diversity, fixed hash tie-break, no image cherry-picking."""
    pool = sorted(pool, key=lambda r: (r['score'], r[ID]))
    chosen = [r for r in pool if r[ID] in anchors][:n]
    bins = np.array_split(np.arange(len(pool)), n)
    for indices in bins:
        if len(chosen) == n: break
        candidates = [pool[int(i)] for i in indices if pool[int(i)] not in chosen]
        if not candidates: candidates = [r for r in pool if r not in chosen]
        counts = collections.Counter(r['category_name'] for r in chosen)
        chosen.append(min(candidates, key=lambda r: (counts[r['category_name']], key(r))))
    assert len(chosen) == n
    return chosen

def query_fields(r):
    # Explicit whitelist: validation truth, grade, reason and scores never enter matching.
    return {k: r[k] for k in [ID, 'category_name', 'category_reason', '版别', *SIDES]}

@functools.lru_cache(maxsize=850)
def vector(cid):
    p = EXP / 'A5/特征缓存' / (cid + '.npy')
    if not p.exists(): return None
    z = np.load(p).astype('float32')[:, VALID].mean(1)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)

@functools.lru_cache(maxsize=240)
def appearance(front, back):
    # Small copies used ONLY for ranking; the displayed assets remain original 1024 px files.
    out = []
    yy, xx = np.mgrid[:24, :24]; mask = (xx-11.5)**2+(yy-11.5)**2 < 10.4**2
    for path in [front, back]:
        with Image.open(path) as im:
            z = np.asarray(im.convert('RGB').resize((24,24), Image.Resampling.BILINEAR), dtype='float32') / 255
        out.append(z[mask].reshape(-1))
    return np.concatenate(out)

def match(q, normals):
    cat = q['category_name']; parts = cat.split(' → ')
    pool = [r for r in normals if r['category_name'] == cat and r['category_reason'] == 'confirmed'] if q['category_reason'] == 'confirmed' else []
    prefix, depth = cat, 0
    if not pool:
        for n in range(len(parts)-1, 2, -1):
            prefix = ' → '.join(parts[:n])
            pool = [r for r in normals if r['category_reason'] == 'confirmed' and (r['category_name'] == prefix or r['category_name'].startswith(prefix+' → '))]
            if pool: depth = len(parts)-n; break
    if not pool: raise ValueError('No appropriate category reference; do not silently use global bank: '+q[ID])
    pool = [r for r in pool if r[ID] != q[ID]]
    # Existing learned descriptors rank cached candidates; exact variety has priority.
    qv = vector(q[ID]); ranked = []
    for r in pool:
        rv = vector(r[ID]); sim = float((qv*rv).sum(1).mean()) if qv is not None and rv is not None else None
        ranked.append((r, sim, r['版别'] == q['版别']))
    ranked.sort(key=lambda t: (not t[2], t[1] is None, -(t[1] or 0), key(t[0])))
    candidates = ranked[:6]  # bounded image inspection/ranking, never scan all normal photos
    qa = appearance(q[SIDES[0]], q[SIDES[1]])
    scored = []
    for r, sim, exact in candidates:
        diff = float(np.mean(np.abs(qa-appearance(r[SIDES[0]],r[SIDES[1]]))))
        scored.append((r, sim, exact, diff))
    scored.sort(key=lambda t: (not t[2], .7*(1-t[1])+.3*t[3] if t[1] is not None else 1+t[3], key(t[0])))
    return scored[:2], {'prefix':prefix,'fallback_depth':depth,'pool_count':len(pool),'shortlist_count':len(candidates),
        'cached_pool_count':sum(t[1] is not None for t in ranked),
        'method':'已知类别优先；版别字符串优先；已有A5双面池化特征余弦相似度筛至最多6枚，再结合双面低频RGB外观距离取2枚。未用目标真实品相；不保证品相或拍摄光照一致。'}

def copy_coin(r):
    assets = []; cid = r[ID]
    for side, col in enumerate(SIDES):
        src = Path(r[col]); dst = ROOT/'图片'/f'{cid}_{side}{src.suffix.lower()}'
        with Image.open(src) as im: size = list(im.size)
        assert size == [1024,1024], (cid, size)
        if dst.exists(): assert sha(dst) == sha(src), 'Refusing to overwrite different image'
        else: shutil.copy2(src, dst)
        assert sha(dst) == sha(src)
        ASSETS[str(dst.relative_to(ROOT)).replace('\\','/')] = {'source':str(src), 'sha256':sha(dst),'size':size,'id':cid,'side':side}
        assets.append(str(dst.relative_to(ROOT)).replace('\\','/'))
    return {'id':cid,'certificate':r['证书编号'],'split':r['split'],'category':r['category_name'], 'variety':r['版别'],
            'images':assets,'truth':{'label':int(r['二分类标签']),'grade':r['原始评级'],'grade_group':r['grade_group'],'code':r['异常代码']}}

if __name__ == '__main__':
    for name in ['图片','辅助热图','结果']: (ROOT/name).mkdir(exist_ok=True)
    cfgpath=EXP/'A1/融合对照/A0_A1集成/frozen_ensemble_config.json'
    cfg=json.loads(cfgpath.read_text(encoding='utf8')); INPUTS[str(cfgpath)]=sha(cfgpath)
    threshold=cfg['threshold']
    val=read('A5/结果/val_manifest.csv'); train=read('A3/结果/train_manifest.csv')
    cats={s:{r[ID]:r for r in read(f'A2/结果/{s}_manifest.csv')} for s in ['train','val']}
    pred={r[ID]:r for r in read('A1/融合对照/A0_A1集成/val_predictions.csv')}
    for split,rows in [('train',train),('val',val)]:
        for r in rows:
            r.update({k:cats[split][r[ID]][k] for k in ['category_name','category_reason','版别']}); r['split']=split
            if split=='val':
                p=pred[r[ID]]; r['score']=float(p['ensemble_score']); r['pred']=int(p['ensemble_pred'])
                assert r['pred']==int(r['score']>=threshold) and float(p['threshold'])==threshold
    normals=[r for r in train if r['二分类标签']=='0']
    buckets={
      '正常AU/XF误报':[r for r in val if r['二分类标签']=='0' and r['grade_group'] in ['AU','XF'] and r['pred']==1],
      '92清洗漏检':[r for r in val if r['二分类标签']=='1' and r['异常代码']=='92' and r['pred']==0],
      '92清洗检出':[r for r in val if r['二分类标签']=='1' and r['异常代码']=='92' and r['pred']==1],
      '正常AU/XF正确':[r for r in val if r['二分类标签']=='0' and r['grade_group'] in ['AU','XF'] and r['pred']==0],
      '其他异常':[r for r in val if r['二分类标签']=='1' and r['异常代码'] in ['97','98']]}
    selected=[]
    for label,n in [('正常AU/XF误报',10),('92清洗漏检',10),('92清洗检出',4),('正常AU/XF正确',4),('其他异常',2)]:
        pool=buckets[label]
        if label.startswith('正常'):
            picked=sum([choose([r for r in pool if r['grade_group']==g],n//2) for g in ['AU','XF']],[])
        elif label=='其他异常': picked=sum([choose([r for r in pool if r['异常代码']==g],1) for g in ['97','98']],[])
        else: picked=choose(pool,n)
        for r in picked: r['review_group']=label
        selected.extend(picked)
    selected.sort(key=key)  # hide group order in blind review
    assert len(selected)==30 and len({r[ID] for r in selected})==30
    VALID=np.load(EXP/'A5/结果/valid_patch_indices.npy')
    maps=np.load(EXP/'A5/结果/local_maps.npz')['distances']
    # Identical scale to A5 evaluate_visualize.py, no retraining or local score recomputation.
    vmax=float(np.quantile(maps,.99)); indices={r[ID]:i for i,r in enumerate(val)}
    a5plans=json.loads((EXP/'A5/结果/reference_plan.json').read_text(encoding='utf8'))['plans']
    ASSETS={}; samples=[]; refrows=[]
    for i,r in enumerate(selected):
        refs,meta=match(query_fields(r),normals); item=copy_coin(r)
        item.update({'review_id':f'{i+1:02d}','score':r['score'],'prediction':r['pred'],'group':r['review_group'],'references':[], 'match':meta})
        for rank,(ref,sim,exact,diff) in enumerate(refs,1):
            ri=copy_coin(ref); ri.update({'feature_similarity':sim,'appearance_distance':diff,'same_variety':exact})
            item['references'].append(ri)
            refrows.append({'target_certificate':r['证书编号'],'target_split':'val','reference_certificate':ref['证书编号'],'reference_split':'train','rank':rank,'target_category':r['category_name'],'reference_category':ref['category_name'],'same_variety':exact,'fallback_depth':meta['fallback_depth'],'prefix':meta['prefix'],'pool_count':meta['pool_count'],'feature_similarity':sim,'appearance_distance':diff,'reference_grade':ref['原始评级']})
        item['heatmaps']=[]
        oldplan=a5plans[r['category_name']+'|'+r['category_reason']]
        item['heatmap_reference']={'prefix':oldplan['prefix'],'fallback_depth':oldplan['fallback_depth'],'reference_count':32}
        for side in [0,1]:
            heat=np.zeros((576,4),dtype='uint8'); values=np.clip(maps[indices[r[ID]],1,side]/vmax,0,1)
            heat[VALID,0]=255; heat[VALID,1]=(220*(1-values)).astype('uint8'); heat[VALID,2]=20; heat[VALID,3]=(values*165).astype('uint8')
            dest=ROOT/'辅助热图'/f'{r[ID]}_{side}.png'
            Image.fromarray(heat.reshape(24,24,4)).resize((1024,1024),Image.Resampling.NEAREST).save(dest)
            item['heatmaps'].append(str(dest.relative_to(ROOT)).replace('\\','/'))
        samples.append(item)
        print('prepared',i+1,r[ID],len(refs),'fallback',meta['fallback_depth'],flush=True)
    dataset_id=hashlib.sha256(json.dumps(samples,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:16]
    payload={'version':'A6-1','dataset_id':dataset_id,'date':'2026-11-14','threshold':threshold,'heatmap_max':vmax,'samples':samples}
    save(ROOT/'结果/review_manifest.json',payload)
    writecsv(ROOT/'结果/样本清单.csv',[{'review_id':s['review_id'],'certificate':s['certificate'],'split':s['split'],'group':s['group'],'grade':s['truth']['grade'],'code':s['truth']['code'],'category':s['category'],'score':s['score'],'threshold':threshold,'prediction':s['prediction']} for s in samples])
    writecsv(ROOT/'结果/参照匹配.csv',refrows)
    save(ROOT/'结果/图像来源.json',ASSETS)
    empty=[{'review_id':s['review_id'],'certificate':s['certificate'],'split':'val','status':'未观察','part':'','notes':'','localizable':'未填写','reference_id':s['references'][0]['certificate'],'label_seen':False,'heatmap_seen':False,'updated_at':''} for s in samples]
    save(ROOT/'结果/人工观察_空白模板.json',{'dataset_id':dataset_id,'reviews':empty})
    writecsv(ROOT/'结果/人工观察_空白模板.csv',empty)
    stats={'target_count':len(samples),'target_split':'val','groups':dict(collections.Counter(s['group'] for s in samples)),
      'available_group_counts':{k:len(v) for k,v in buckets.items()},'references_unique':len({r['id'] for s in samples for r in s['references']}),
      'reference_pairs':len(refrows),'fallback_targets':sum(s['match']['fallback_depth']>0 for s in samples),'different_variety_pairs':sum(not r['same_variety'] for s in samples for r in s['references']),
      'reference_grades':dict(collections.Counter(r['truth']['grade_group'] for s in samples for r in s['references'])),
      'copied_images':len(ASSETS),'image_bytes':sum((ROOT/k).stat().st_size for k in ASSETS),'threshold':threshold,'dataset_id':dataset_id}
    save(ROOT/'结果/准备统计.json',stats)
    save(ROOT/'结果/输入文件.json',INPUTS)
    template=(ROOT/'代码/template.html').read_text(encoding='utf8')
    html=template.replace('/*__DATA__*/', 'const DATA = '+json.dumps(payload,ensure_ascii=False).replace('</','<\\/')+';')
    (ROOT/'人工观察.html').write_text(html,encoding='utf8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))
