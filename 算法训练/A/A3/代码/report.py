from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'结果'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def table(rs,keys):
 def fmt(x):return f'{x:.6f}' if isinstance(x,float) else str(x)
 return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+'\n'.join('| '+' | '.join(fmt(r.get(k,'')) for k in keys)+' |' for r in rs)+'\n'
e=js(O/'evaluation.json');h=js(O/'history.json');audit=js(O/'grade_audit.json');fr=js(R/'模型/frozen_evaluation.json');done=js(O/'TRAIN_COMPLETE.json');trans=read(O/'error_transitions.csv')
rows=e['comparison'];get=lambda s,n:next(r for r in rows if r['split']==s and r['model']==n)
summary=[]
for s in ['val','test']:
 for n in ['A3','A0+A3']:
  for lab in ['0','1']:
   rr=[r for r in trans if r['split']==s and r['model']==n and r['label']==lab]
   summary.append({'split':s,'model':n,'label':lab,**{k:sum(r['transition']==k for r in rr) for k in ['corrected','introduced','both_wrong']}})
a1=get('test','A1');a3=get('test','A3');v1=get('val','A1');v3=get('val','A3');ens=get('test','A0+A3');old=get('test','A0+A1')
text='# A3 正常品相辅助学习实验报告\n\n正式项目：E:\\异常检测；本轮输出仅在算法训练/A3。\n\n'
text+='## 方法与公平性\n\nA3 保留 A1 共享 ConvNeXt-Tiny、每面768维、双面拼接1536维及1536→256→1 GELU异常头。在同一1536维图像特征上增加 Linear(1536,4) 品相头。全参数训练，BCE + 0.2 × 有效正常样本平均交叉熵；批次无有效品相标签时辅助损失为零。异常头、骨干初始化与 A1 逐张量相等；两个任务对共享骨干梯度均非零，见 model_verification.json。推理 forward 仅接收双面图像，不接收品相或正常/异常标签。\n\n'
text+='沿用 ImageNet 初始化（未载入 A1 best）、seed42、batch8、384分辨率、独立±10°旋转、AdamW 主干2e-5/头1e-4、WD0.01、AMP、梯度裁剪5。15–50轮、patience10，以验证异常AUROC选最佳，第6轮起沿用原ReduceLROnPlateau。复用SSD原图缓存，无重复预处理。每轮采样和增强种子与A1相同；实际早停轮数由共同规则决定，不人为延长A3。A1完成15轮、最佳4轮。\n\n'
text+=f'A3 完成 {done["epochs"]} 轮，最佳第 {done["best_epoch"]} 轮，训练/验证/保存累计 {done["training_seconds"]/60:.2f} 分钟。仅运行 λ=0.2 一次完整实验；没有进行测试集调参或超参数搜索。A2实际是推理时输入真实类别的条件模型，不是类别辅助损失，其机制与A3不同。\n\n'
text+='## 标签检查\n\n只读现有Excel“钱币详细信息”的证书号、评级，按证书号核对既有划分。保留整币配对与原6033/755/755划分，证书跨集合无交集，原CSV哈希一致。仅使用正常、无问题代码、数字等级合法且Excel评级唯一一致的标签。VF20–35与极少的G4/F12/F15合并为“VF及以下”；XF40/45、AU50/53/55/58、MS60–70分别成组。分组依据训练集；SP/PR代表不同铸造类型且样本稀少，首版不强行映射为MS。92/97/98等一律不作为品相等级。未监督样本均保留二分类训练。\n\n'
text+=table([{'split':s,**{k:v for k,v in a.items() if k in ['total','normal','supervised']},**a['groups']} for s,a in audit.items()],['split','total','normal','supervised','VF及以下','XF','AU','MS','未监督'])
text+='\n完整数字等级分布见 numeric_grade_distribution.csv；逐币标签、排除理由见 train/val/test_manifest.csv。训练未监督的20枚正常币均为SP/PR；G04与G4按同一数值4匹配，合并到VF及以下；没有推测或补全缺失标签。品相等级不等于异常：上述分支只辅助正常币视觉差异学习，不改变任何二分类标签。\n\n'
text+=f'最佳模型验证品相准确率 {js(O/"grade_validation.json")["accuracy"]:.4f}，测试 {js(O/"grade_test.json")["accuracy"]:.4f}；仅作辅助诊断，不用于早停或选模。\n\n'
text+='## 选择协议与统一指标\n\n单模型阈值最大化验证F1，并列选最高阈值。A0+A3沿用概率加权策略，在预设A0权重0/0.1/0.2/0.3/0.5/1中按验证AUROC、AP、较小权重选择，再选验证F1阈值。先写 frozen_evaluation.json 固化模型哈希、阈值、融合权重和召回对照阈值，之后才首次推理测试集。各旧实验保留原验证阈值。\n\n'
text+=f'A0+A3选择A0权重 {fr["ensemble"]["alpha"]}，阈值 {fr["ensemble"]["threshold"]:.10f}，验证AUROC {fr["ensemble"]["auroc"]:.6f}。即使单模型不领先，也将该融合明确视为验证集互补性对照，不根据测试成绩重新选择方案。\n\n'
text+=table(rows,['split','model','auroc','ap','precision','recall','f1','tp','fp','fn','tn'])
text+='\n## 正常品相误报\n\n'+table(e['normal_grade_fp'],['split','model','grade','n','fp','fpr'])
text+='\n## 相近召回要求对照\n\n仅在验证集选择满足70%、75%、80%、85%、90%以及A1原工作点召回要求的最高阈值，冻结后用于测试。测试实际召回可能不同，因此必须同时看FP和FN，不能将验证召回相同说成测试召回完全相同。未按测试标签另选阈值。\n\n'
text+=table(e['matched_recall'],['split','model','target','threshold','recall','fp','fn','f1'])
text+='\n## 错误迁移与实例\n\n以下均相对A1：label0代表正常误报，label1代表异常漏检。逐币全量结果见 error_transitions.csv；仅作冻结方案解释。\n\n'+table(summary,['split','model','label','corrected','introduced','both_wrong'])
for lab,tr,title in [('0','corrected','成功纠正的正常币'),('0','introduced','新增正常误报'),('1','introduced','新增漏检的异常币'),('1','corrected','成功检出的原漏检异常币')]:
 rr=[r for r in trans if r['split']=='test' and r['model']=='A3' and r['label']==lab and r['transition']==tr]
 text+=f'\n### {title}（共{len(rr)}枚，按编号展示前10枚）\n\n'+table(rr[:10],['钱币编号','原始评级','异常代码','grade','A1_score','new_score'])
text+='\n## 结论\n\n'
text+=f'A3相对A1：验证AUROC变化 {v3["auroc"]-v1["auroc"]:+.6f}，测试AUROC变化 {a3["auroc"]-a1["auroc"]:+.6f}，AP变化 {a3["ap"]-a1["ap"]:+.6f}；正常误报FP {a1["fp"]}→{a3["fp"]}，异常漏检FN {a1["fn"]}→{a3["fn"]}。A0+A3相对A0+A1：FP {old["fp"]}→{ens["fp"]}、FN {old["fn"]}→{ens["fn"]}、AUROC {old["auroc"]:.6f}→{ens["auroc"]:.6f}。\n\n'
support=v3['auroc']>v1['auroc'] and a3['auroc']>a1['auroc'] and a3['fp']<a1['fp'] and a3['fn']<=a1['fn']
text+=('本次结果对“学习正常品相变化能够改善钱币异常检测”提供初步支持，但只限单种子固定划分，不能视为统计显著或普遍成立。' if support else '本次实验不足以支持“学习正常品相变化能够改善钱币异常检测”这一假设；需要区分排序改善、阈值取舍与融合互补，不能以个别指标或品相分类准确率证明异常检测改善。')
text+='最有价值的下一步是在保持测试集封闭的条件下，用多个配对随机种子复核A1与A3，先确认收益/退化的稳定性，再考虑小范围辅助权重调整。当前实验不追加训练，不以测试AU/XF错误反向挑超参数。\n\n'
text+='![训练曲线](结果/training_curves.png)\n\n![正常分组误报](结果/grade_fp.png)\n\n![相近召回对照](结果/recall_fp.png)\n'
(R/'实验报告.md').write_text(text,encoding='utf-8')
fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].plot([r['epoch'] for r in h],[r['loss'] for r in h]);ax[0].set(xlabel='Epoch',ylabel='BCE + 0.2 CE')
for k in ['auroc','ap']:ax[1].plot([r['epoch'] for r in h],[r['val_'+k] for r in h],label=k)
ax[1].legend();ax[1].set(xlabel='Epoch',ylabel='Validation');fig.tight_layout();fig.savefig(O/'training_curves.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4));gs=['VF及以下','XF','AU','MS'];x=np.arange(4)
for j,n in enumerate(['A1','A3','A0+A1','A0+A3']):ax.bar(x+(j-1.5)*.2,[next(r['fp'] for r in e['normal_grade_fp'] if r['split']=='test' and r['model']==n and r['grade']==g) for g in gs],width=.2,label=n)
ax.set(xticks=x,xticklabels=['VF and lower','XF','AU','MS'],ylabel='False positives (test)');ax.legend();fig.tight_layout();fig.savefig(O/'grade_fp.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for ax,s in zip(axs,['val','test']):
 for n in ['A1','A3']:
  rr=[r for r in e['matched_recall'] if r['split']==s and r['model']==n];ax.plot([r['recall'] for r in rr],[r['fp'] for r in rr],'o-',label=n)
 ax.set(xlabel='Actual anomaly recall',ylabel='Normal false positives',title=s+' (thresholds selected on val)');ax.legend()
fig.tight_layout();fig.savefig(O/'recall_fp.png',dpi=160);plt.close(fig)
print(text[text.index('## 结论'):])
