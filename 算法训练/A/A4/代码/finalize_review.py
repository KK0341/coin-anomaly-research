import json,hashlib,py_compile,time
from pathlib import Path
R=Path(r'E:\异常检测\算法训练\A4')
p=R/'代码/interpret.py';s=p.read_text(encoding='utf-8');old="p=R/'实验报告.md';p.write_text(p.read_text(encoding='utf-8').replace('<!-- INTERPRETATION -->',text),encoding='utf-8')"
new="p=R/'实验报告.md'\n if p.exists():\n  doc=p.read_text(encoding='utf-8')\n  if '<!-- INTERPRETATION -->' in doc:doc=doc.replace('<!-- INTERPRETATION -->',text)\n  else:\n   start=doc.index('## 结果解释与明确结论')+len('## 结果解释与明确结论');end=doc.index('![训练曲线',start);doc=doc[:start]+'\\n\\n'+text+'\\n\\n'+doc[end:]\n  p.write_text(doc,encoding='utf-8')"
assert old in s;s=s.replace(old,new);p.write_text(s,encoding='utf-8')
for p in R.rglob('*.py'):py_compile.compile(str(p),doraise=True)
doc=(R/'实验报告.md').read_text(encoding='utf-8');assert '<!-- INTERPRETATION -->' not in doc;assert doc.rfind('## 后续专家知识引导路线')>doc.rfind('## 文件与复现')
for n in ['A4a','A4b']:
 for f in ['模型/best.pt','模型/last.pt','模型/config.json','模型/threshold.json','结果/val_predictions.csv','结果/test_predictions.csv','代码/train.py']:assert (R/n/f).stat().st_size>0
v=json.loads((R/'结果/final_verification.json').read_text(encoding='utf-8'));assert v['historical_files_unchanged']==1109
summary={'completed_unix':time.time(),'experiments':['A4a','A4b'],'seed':42,'report':'实验报告.md','report_sha256':hashlib.sha256((R/'实验报告.md').read_bytes()).hexdigest(),'report_reviewed':True,'figure_reviewed':True,'historical_files_unchanged':1109,'multiseed_added':False,'test_policy':'exploratory frozen evaluation','conclusion':'No stable overall incremental benefit; limited A4a signals; prioritize category-specific normal local references and expert evidence verification.'}
(R/'结果/COMPLETE.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(R/'结果/PIPELINE_COMPLETE.json').write_text(json.dumps({'completed_unix':time.time(),'note':'Artifacts and final report reviewed; see COMPLETE.json'},indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False))
