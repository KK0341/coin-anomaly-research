from pathlib import Path
import csv,json,re,hashlib,collections
R=Path(__file__).resolve().parents[1]; A1=R.parent/'A1'; S=R.parents[1]/'数据划分'
def js(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def wc(p,rs):
 with p.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
cfg=json.loads((A1/'模型/config.json').read_text(encoding='utf-8'))
raw={s:list(csv.DictReader((S/f).open(encoding='utf-8-sig',newline=''))) for s,f in {'train':'训练集.csv','val':'验证集.csv','test':'测试集.csv'}.items()}
excel=json.loads((R/'结果/excel_grade_rows.json').read_text(encoding='utf-8')); ex=collections.defaultdict(set)
for r in excel:
 cert=str(r['证书号']).removesuffix('.0').zfill(8);ex[cert].add(str(r['评级']).strip())
names=['VF及以下','XF','AU','MS']; stats={}; allids=set(); distribution=[]
for s,rs in raw.items():
 out=[]; ids={r['证书编号'] for r in rs}; assert len(ids)==len(rs) and not ids&allids;allids|=ids
 for r in rs:
  g=r['正常数字等级']; match=re.fullmatch(r'(G|F|VF|XF|AU|MS)(\d+)',g); idx=-1;reason='abnormal' if r['二分类标签']=='1' else 'unsupported_or_missing_grade'
  if r['二分类标签']=='0' and match and not r['异常代码']:
   prefix,num=match[1],int(match[2]); valid={'G':{4,6},'F':{12,15},'VF':{20,25,30,35},'XF':{40,45},'AU':{50,53,55,58},'MS':set(range(60,71))}
   original=re.match(r'^(G|F|VF|XF|AU|MS)(\d+)',r['原始评级'])
   reliable=ex[r['证书编号']]=={r['原始评级']} and num in valid[prefix] and original is not None and original[1]==prefix and int(original[2])==num
   if reliable:idx=names.index('VF及以下' if num<=35 else prefix);reason='verified_excel_numeric'
   else:reason='excel_conflict_or_invalid'
  out.append({**{k:r[k] for k in ['钱币编号','证书编号','正面路径','背面路径','二分类标签','原始评级','异常代码','正常数字等级']},'grade_id':idx,'grade_group':names[idx] if idx>=0 else '未监督','grade_reason':reason})
 wc(R/'结果'/f'{s}_manifest.csv',out)
 stats[s]={'total':len(rs),'normal':sum(r['二分类标签']=='0' for r in rs),'supervised':sum(r['grade_id']>=0 for r in out),'groups':dict(collections.Counter(r['grade_group'] for r in out if r['二分类标签']=='0')),'reasons':dict(collections.Counter(r['grade_reason'] for r in out))}
 for grade,n in sorted(collections.Counter(r['正常数字等级'] for r in rs if r['二分类标签']=='0').items()):distribution.append(dict(split=s,grade=grade,n=n))
wc(R/'结果/numeric_grade_distribution.csv',distribution);js(R/'结果/grade_audit.json',stats);js(R/'模型/grade_mapping.json',dict(enumerate(names)))
for k in ['training_start_unix_estimate','timing_start_method']:cfg.pop(k,None)
cfg.update(grade_lambda=.2,grade_classes=names,architecture='A1 unchanged anomaly head; parallel Linear(1536,4) grade head on shared image features',grade_loss='mean CE over reliable normal grades in each batch; zero when empty',selection='validation AUROC only',head_parameters='A1 MLP + grade Linear',test_policy='separate evaluation only after frozen validation selection')
js(R/'模型/config.json',cfg);(R/'结果/loader_selection.json').write_bytes((A1/'结果/loader_selection.json').read_bytes())
src=(A1/'代码/train.py').read_text(encoding='utf-8'); start=src.index('def rows(split):');end=src.index('\nclass Coins',start)
src=src[:start]+'''def rows(split):
    with open(OUT/f'{split}_manifest.csv',encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))
'''+src[end:]
src=src.replace('self.head=nn.Sequential(nn.Linear(1536,256),nn.GELU(),nn.Linear(256,1))','self.head=nn.Sequential(nn.Linear(1536,256),nn.GELU(),nn.Linear(256,1))\n        self.grade_head=nn.Linear(1536,4)')
src=src.replace('return z,z[:,None]', 'return z,self.grade_head(features)')
src=src.replace("'params':m.head.parameters()","'params':list(m.head.parameters())+list(m.grade_head.parameters())")
src=src.replace('for x_cpu,y_cpu,_ in batches:', 'for x_cpu,y_cpu,indices in batches:')
src=src.replace('z,_=m(x); loss=nn.functional.binary_cross_entropy_with_logits(z,y)', '''z,g=m(x); anomaly_loss=nn.functional.binary_cross_entropy_with_logits(z,y)
                    grade=torch.tensor([int(dl.dataset.rows[int(i)]['grade_id']) for i in indices],device='cuda')
                    mask=(y==0)&(grade>=0)
                    grade_loss=nn.functional.cross_entropy(g[mask],grade[mask]) if mask.any() else g.sum()*0
                    loss=anomaly_loss+0.2*grade_loss''')
start=src.index('def finish(');end=src.index('\ndef train():',start)
src=src[:start]+'''def finish(m,config,history):
    best=torch.load(MODEL/'best.pt',map_location='cpu',weights_only=False); m.load_state_dict(best['model'])
    val=Coins('val'); values=predict(m,loader(val,config['batch_coins'],workers=0))
    precision,recall,thresholds=precision_recall_curve(values[0],values[1]); fs=2*precision[:-1]*recall[:-1]/(precision[:-1]+recall[:-1]+1e-12)
    threshold=float(thresholds[np.flatnonzero(fs==fs.max())[-1]])
    save_json(MODEL/'threshold.json',{'threshold':threshold,'source':'validation only','best_epoch':best['epoch'],'val_metrics':metrics(values[0],values[1],threshold)})
    write_predictions('val',val,values,threshold)
    grade=np.array([int(r['grade_id']) for r in val.rows]);mask=grade>=0
    save_json(OUT/'grade_validation.json',{'n':int(mask.sum()),'accuracy':float((values[4].argmax(1)[mask]==grade[mask]).mean()),'confusion_matrix':confusion_matrix(grade[mask],values[4].argmax(1)[mask],labels=list(range(4))).tolist()})
    save_json(OUT/'TRAIN_COMPLETE.json',{'epochs':len(history),'best_epoch':best['epoch'],'val_auroc':best['val_auroc'],'training_seconds':sum(h['epoch_total_seconds'] for h in history)})
    print('TRAIN_COMPLETE',flush=True)
'''+src[end:]
src=src.replace('    finish(m,config,history)','    del dl,vl,ds,val,opt,scaler; gc.collect(); torch.cuda.empty_cache()\n    finish(m,config,history)')
(R/'代码/train.py').write_text(src,encoding='utf-8')
paths=[p for d in ['A0','A1','A2'] for p in (R.parent/d).rglob('*') if p.is_file() and p.suffix in ['.py','.md','.json','.csv','.pt']]
js(R/'结果/source_hashes.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
print(json.dumps(stats,ensure_ascii=False))
