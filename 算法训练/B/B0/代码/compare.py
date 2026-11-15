import json
from pathlib import Path
R=Path(__file__).resolve().parents[2]
data={n:json.loads((R/n/'test_metrics.json').read_text('utf8')) for n in ['B0','B1']}
lines=['# B0 / B1 第一轮对比','','两组均已在本机 RTX 5060 Laptop GPU 上依次完成，远程未进行正式训练。以下为198枚真实异常测试币直接进入六分类模型的结果。','','|指标|B0|B1|B0−B1（百分点）|','|---|---:|---:|---:|']
for k in ['accuracy','macro_f1','balanced_accuracy']:
    a,b=data['B0'][k],data['B1'][k]; lines.append(f'|{k}|{a:.4%}|{b:.4%}|{(a-b)*100:+.3f}|')
for k,v in data['B0']['per_class'].items():
    a,b=v['f1'],data['B1']['per_class'][k]['f1']; lines.append(f'|{k} {v["name"]} F1|{a:.4%}|{b:.4%}|{(a-b)*100:+.3f}|')
delta=data['B0']['macro_f1']-data['B1']['macro_f1']
lines+=['',f'以测试Macro-F1比较，阶段一迁移在本次固定划分下{"有帮助" if delta>0 else "未显示收益"}，变化{delta*100:+.3f}个百分点。单次实验不代表统计显著性。']
codes=list(data['B0']['per_class'])
for name,m in data.items():
    cm=m['confusion_matrix']; pairs=sorted([(cm[i][j],codes[i],codes[j]) for i in range(6) for j in range(6) if i!=j],reverse=True)[:5]
    lines+=['',name+' 主要混淆（真实→预测）：'+ '；'.join(f'{a}→{b}：{n}枚' for n,a,b in pairs if n)]
lines+=['','可能原因：清洗、损坏及环境损伤的表面外观可能重叠，少数类训练样本少。抽查错误图中既有深浅包浆、反光差异，也有较高置信度的损坏→清洗误判；不能仅凭照片认定原标签错误，视觉成因仍是解释假设。','','|训练与选模|B0|B1|','|---|---:|---:|']
done={n:json.loads((R/n/'COMPLETE.json').read_text('utf8')) for n in data}
hist={n:json.loads((R/n/'history.json').read_text('utf8')) for n in data}
lines += [f'|完成轮数|{done["B0"]["epochs"]}|{done["B1"]["epochs"]}|',f'|最佳轮数|{done["B0"]["best_epoch"]}|{done["B1"]["best_epoch"]}|',f'|最佳模型验证Macro-F1|{done["B0"]["metrics"]["val"]["macro_f1"]:.2%}|{done["B1"]["metrics"]["val"]["macro_f1"]:.2%}|',f'|训练日志末轮Macro-F1|{hist["B0"][-1]["train_macro_f1"]:.2%}|{hist["B1"][-1]["train_macro_f1"]:.2%}|']
lines+=['','两组后期训练F1继续升高、验证损失上升，存在明显过拟合。最终评价使用验证F1最优权重，不使用末轮权重。训练日志F1为训练过程中增强批次上的统计。','迁移收益有限：Balanced Accuracy仅提升0.264个百分点，94和91的F1略降。恒预测清洗的Accuracy为56.57%、Macro-F1仅12.04%，因此不能仅用Accuracy评价这个不平衡任务。少数类测试集仅8—10枚，单次划分结论不等于统计显著性。','','完整流程评价复用A8单模型已冻结的二分类预测及验证阈值0.3065270483，未运行A0+A8+A5b集成流程。正常币误报也参与以下完整流程六类F1计算。','','|接入A8后的测试指标|B0|B1|','|---|---:|---:|']
flow={n:json.loads((R/n/'test_pipeline_metrics.json').read_text('utf8')) for n in data}
for title,key in [('198枚真实六类异常币端到端正确率','six_abnormal_end_to_end_accuracy'),('完整流程六类Macro-F1（含正常币误报影响）','six_class_macro_f1_in_full_flow'),('已覆盖754枚七分类Accuracy（正常+六异常）','accuracy_7class_covered')]:
    lines.append(f'|{title}|{flow["B0"][key]:.2%}|{flow["B1"][key]:.2%}|')
lines+=['','A8将198枚六类异常币中的71枚漏判为正常（漏检35.86%），另将556枚正常币中的69枚误报为异常。测试集中另有1枚82币缘锉修，被标记为体系未覆盖，未计作任何已覆盖类别的正确预测。82/93原标签与原始清单均保留。','下一轮优先只尝试一项改进：以B0初始化为基础，冻结ConvNeXt前三个stage，仅微调最后一个stage与分类头，其余配置和固定划分不变，检验减少可训练参数能否缓解过拟合。该改进尚未运行，也不保证提升。','核验：从验证/测试逐币预测重新计算Accuracy、Macro-F1、Balanced Accuracy、各类P/R/F1及混淆矩阵，与保存结果一致；两组训练配置除实验名称、初始化和输出路径外一致，公共训练代码哈希一致；最佳轮次等于验证Macro-F1最高轮次。本次汇总未重新推理测试集、未调整模型。']
for name in ['B0','B1']: (R/name/'B0_B1对比.md').write_text('\n'.join(lines),'utf8')
print('\n'.join(lines))
