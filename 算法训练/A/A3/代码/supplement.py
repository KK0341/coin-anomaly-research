"""Describe frozen outputs; never choose new thresholds or train models."""
import csv,json,collections
import numpy as np
from pathlib import Path
from evaluate import load
import train as t
R=t.ROOT;O=t.OUT
e=json.loads((O/'evaluation.json').read_text(encoding='utf-8'));fr=json.loads((t.MODEL/'frozen_evaluation.json').read_text(encoding='utf-8'))
t.save_json(O/'test_metrics.json',next(r for r in e['comparison'] if r['split']=='test' and r['model']=='A3'))
t.save_json(R/'融合/frozen_config.json',{'frozen_time_unix':fr['time_unix'],'formula':'alpha*A0+(1-alpha)*A3','selected':fr['ensemble'],'checkpoint_sha256':fr['checkpoint_sha256']})
t.save_json(R/'融合/metrics.json',[r for r in e['comparison'] if r['model']=='A0+A3'])
groups=[]
for split in ['val','test']:
 ds=t.rows(split)
 for n in ['A1','A3']:
  y,p=load(n,split)
  for rr in fr['matched_recall']:
   if rr['model']!=n:continue
   for g in ['VF及以下','XF','AU','MS','未监督']:
    mask=np.array([r['grade_group']==g and r['二分类标签']=='0' for r in ds]);count=int(mask.sum());fp=int((p[mask]>=rr['threshold']).sum())
    groups.append(dict(split=split,model=n,target=rr['target'],grade=g,n=count,fp=fp,fpr=fp/count if count else None))
t.write_csv(O/'matched_recall_grade_fp.csv',groups)
def tab(rs,cols):return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(str(r[c]) for c in cols)+' |' for r in rs)+'\n'
report=R/'实验报告.md';txt=report.read_text(encoding='utf-8');start=txt.index('本次实验不足以支持');end=txt.index('\n\n![训练曲线]',start)
txt=txt[:start]+'''**结论：本次实验对“学习正常品相变化能够改善钱币异常检测”提供有限、局部支持，但尚不支持它能稳定降低正常误报且不增加异常漏检。** 验证和测试的AUROC/AP同时提高，表明收益不完全是阈值调整；但主工作点存在明显漏检代价，相近召回要求的结果不一致。

- 默认阈值下，AU误报36→23、XF 27→21、MS 37→7，VF及以下11→11。但异常召回71.86%→64.82%，不能据此认定这些组的视觉混淆已被可靠解决。
- 在A1原工作点对应的77.39%验证召回下，验证FP 102→113；冻结阈值后的测试FP 112→116、FN 56→46。此处改善的是测试召回，并没有降低误报。
- 70%召回要求：验证双方FP均85；测试实际召回69.85%与70.35%接近，A3的FP 93→83、FN 60→59。这是一个较干净的局部改善证据，但未在验证FP上显示优势。
- 75%要求：验证FP 98→108；测试106→110、FN57→46。80%要求：验证FP128→125；测试137→123、FN50→43。两点方向不同，不能只挑有利工作点。
- 85%要求下测试FP193→142伴随FN28→37；90%要求下FP261→252伴随FN15→17，均仍有召回取舍。
- 融合在验证集小幅胜过A0+A1；测试F1 0.6472→0.6519、AUROC 0.8576→0.8620，但总错误数同为157。AU/XF误报分别32→35、23→29，MS 31→25，因此未解决重点AU/XF误报问题。

品相辅助学习值得作为研究方向保留，尚不宜据本次结果替换当前A0+A1方案。最有价值的下一步是预先固定目标召回及评价指标，采用多个配对随机种子复核A1/A3；若继续建模，再仅依据新的训练/验证证据小范围调整λ或辅助头，保持已使用测试集的评估边界。当前不追加训练，也不根据测试结果改参数。
'''+txt[end:]
part='\n## 相同验证召回要求下的品相误报细分\n\n主对照固定为A1原验证召回77.39%；完整6组召回要求见 matched_recall_grade_fp.csv。\n\n'
part+=tab([r for r in groups if r['target']==fr['primary_recall_target']],['split','model','grade','n','fp','fpr'])
txt+=part
txt+='''
## 运行记录与复现

本轮正式训练约33.62分钟，早停于15轮，最佳第4轮。初次启动尚未完成第1轮时发现G04/G4前导零格式误排除，已停止该启动并修正后从原初始化重新训练；初次日志单独保留在“结果/初次启动_前导零格式修正”，不计入正式15轮，也未依据验证或测试成绩选择重启。

代码/train.py --mode train 可按last.pt恢复完整轮次；已完成实验无需重训。代码/evaluate.py validation只允许测试评估前运行；随后test阶段复用已保存测试预测。代码/report.py与supplement.py顺序执行可重建报告；verify_results.py核验旧文件哈希、划分、最佳轮次、优化器步数和最佳模型验证重载一致性。Python路径见模型/config.json。原缓存、所有原实验模型与A3 best/last均保留。
'''
if (O/'verification.json').exists():
 v=json.loads((O/'verification.json').read_text(encoding='utf-8'))
 txt+='\n## 最终核验\n\n'+f'历史A0/A1/A2的{v["historical_files_unchanged"]}个已记录文件SHA256未变；划分哈希未变。实际优化器步数{v["optimizer_step"]}，等于15×755批；最佳骨干有{v["changed_backbone_tensors"]}个张量不同于ImageNet初始化。最佳模型重载后的验证概率最大绝对误差为{v["validation_reload_max_abs_difference"]}。测试预测在验证选择冻结后生成。完整记录见结果/verification.json。\n'
report.write_text(txt,encoding='utf-8')
print(tab([r for r in groups if r['split']=='test' and r['target']==fr['primary_recall_target']],['model','grade','n','fp']))
