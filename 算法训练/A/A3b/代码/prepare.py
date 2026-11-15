"""Read-only Excel extraction and minimal A3b preparation. No test predictions read."""
from pathlib import Path
import csv,json,re,hashlib,collections,difflib,sys
import openpyxl
R=Path(__file__).resolve().parents[1]; A=R.parent/'A3'; O=R/'结果'; M=R/'模型'
def js(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def wc(p,rs):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def sha(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
assert not (M/'last.pt').exists(), 'Do not prepare over a started run'
source=R.parents[1]/'袁世凯采集记录.xlsx'
w=openpyxl.load_workbook(source,read_only=True,data_only=True); it=w['钱币详细信息'].iter_rows(values_only=True);head=next(it)
ex=collections.defaultdict(set)
for row in it:
 d=dict(zip(head,row));ex[str(d['证书号']).removesuffix('.0').zfill(8)].add(str(d['评级']).strip())
w.close()
names=['VF及以下','XF','AU','MS'];stats={};dist=[]; added=[];ids=set()
for split in ['train','val','test']:
 old=read(A/'结果'/f'{split}_manifest.csv');out=[]
 for r in old:
  assert r['钱币编号'] not in ids;ids.add(r['钱币编号']); q=dict(r)
  if r['二分类标签']=='1':
   raw=r['原始评级']; match=re.fullmatch(r'Genuine\s*-\s*(PO|FR|AG|G|VG|F|VF|XF|AU|MS|UNC)\s+Details\s*\(\d+\s*-\s*[^()]+\)',raw,re.I)
   q.update(grade_id='-1',grade_group='未监督',grade_reason='missing_or_unsupported_details')
   if ex[r['证书编号']]!={raw}:q['grade_reason']='excel_conflict_or_missing'
   elif match:
    token=match[1].upper()
    if token in ['UNC','MS']:q['grade_reason']='unc_or_ms_details_not_numeric_ms'
    else:
     group=token if token in ['XF','AU'] else names[0]
     q.update(grade_id=str(names.index(group)),grade_group=group,grade_reason='verified_excel_explicit_details')
     added.append({'split':split,**q})
  assert all(q[k]==r[k] for k in r if not k.startswith('grade_'))
  if r['二分类标签']=='0':assert q==r
  out.append(q)
 wc(O/f'{split}_manifest.csv',out)
 ab=[r for r in out if r['二分类标签']=='1'];new=[r for r in ab if int(r['grade_id'])>=0]
 stats[split]={'total':len(out),'abnormal':len(ab),'normal_supervised':sum(r['二分类标签']=='0' and int(r['grade_id'])>=0 for r in out),'new_abnormal_supervised':len(new),'abnormal_excluded_reasons':dict(collections.Counter(r['grade_reason'] for r in ab if int(r['grade_id'])<0))}
 for (group,code),n in sorted(collections.Counter((r['grade_group'],r['异常代码']) for r in new).items()):dist.append(dict(split=split,grade_group=group,abnormal_code=code,n=n))
wc(O/'added_abnormal_grade_rows.csv',added);wc(O/'added_abnormal_distribution.csv',dist);js(O/'grade_summary.json',stats)
cfg=json.loads((A/'模型/config.json').read_text(encoding='utf-8'))
for k in ['training_start_unix_estimate','timing_start_method']:cfg.pop(k,None)
cfg.update(experiment='A3b',grade_loss='mean CE over all reliable normal and abnormal grades per batch; zero when empty',test_policy='exploratory frozen evaluation: hypothesis informed by prior test errors; no test-driven tuning')
js(M/'config.json',cfg)
for sub in ['结果/loader_selection.json','模型/grade_mapping.json']:(R/sub).write_bytes((A/sub).read_bytes())
src=(A/'代码/train.py').read_text(encoding='utf-8');assert src.count('mask=(y==0)&(grade>=0)')==1
dst=src.replace('mask=(y==0)&(grade>=0)','mask=(grade>=0)')
(R/'代码/train.py').write_text(dst,encoding='utf-8');(O/'train_diff.patch').write_text(''.join(difflib.unified_diff(src.splitlines(True),dst.splitlines(True),fromfile='A3/train.py',tofile='A3b/train.py')),encoding='utf-8')
paths=[p for name in ['A0','A1','A2','A3'] for p in (R.parent/name).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
js(O/'historical_hashes.json',{str(p):sha(p) for p in paths})
js(O/'preparation_checks.json',{'excel_sha256':sha(source),'normal_grades_unchanged':True,'binary_labels_pairs_order_unchanged':True,'disjoint_coin_ids':True,'training_diff':'only grade loss mask','python':sys.executable})
js(M/'protocol.json',{'seed':42,'lambda':.2,'model_selection':'validation AUROC, A3 unchanged','threshold_selection':'validation max F1, highest threshold on tie','matched_recall_targets':[.70,.75,.80,.85,.90],'additional_primary_target':'A1 validation recall at its original validation-F1 threshold','matched_rule':'highest validation threshold satisfying target; apply unchanged to test and report achieved recall','extra_seeds':'deferred; this controlled first experiment is seed=42 only','test_boundary':'exploratory because hypothesis was prompted by previous test errors; freeze before one test inference; no retuning','excluded':'UNC Details and MS Details without numeric grades excluded; no abnormal-code-to-grade mapping'})
print(json.dumps(stats,ensure_ascii=False),flush=True)
