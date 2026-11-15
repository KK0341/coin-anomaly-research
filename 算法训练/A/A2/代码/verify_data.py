from pathlib import Path
import csv,json,hashlib,collections
R=Path(__file__).resolve().parents[1];S=Path(r'E:\异常检测\数据划分')
cache=json.loads(Path(r'D:\coin_image_cache\dataset_v1_original\manifest.json').read_text(encoding='utf-8'))
audit={};ids=set();hist=0
for split,filename in [('train','训练集.csv'),('val','验证集.csv'),('test','测试集.csv')]:
 original=list(csv.DictReader((S/filename).open(encoding='utf-8-sig',newline='')))
 derived=list(csv.DictReader((R/'结果'/f'{split}_manifest.csv').open(encoding='utf-8-sig',newline='')))
 assert len(original)==len(derived)
 for a,b in zip(original,derived):
  for k in ['钱币编号','正面路径','背面路径','二分类标签']:assert a[k]==b[k]
  assert a['钱币编号'] not in ids;ids.add(a['钱币编号'])
  if a['本次修正组']=='原96枚历史异常':assert b['category_id']=='0';hist+=1
  for side in ['正面','背面']:
   p=Path(cache[a[side+'路径']]);assert p.is_file()
 audit[split]={'n':len(derived),'labels':dict(collections.Counter(r['二分类标签'] for r in derived))}
assert hist==96
(R/'结果/data_integrity.json').write_text(json.dumps({'passed':True,'historical_96_all_unknown':True,'all_15086_cached_paths_exist':True,'source_rows_preserved':audit},ensure_ascii=False,indent=2),encoding='utf-8')
print('DATA INTEGRITY PASSED',audit)
