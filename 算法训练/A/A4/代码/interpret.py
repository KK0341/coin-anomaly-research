import json,csv
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'结果'
def main():
 e=json.loads((O/'evaluation.json').read_text(encoding='utf-8'));f=json.loads((R/'frozen_evaluation.json').read_text(encoding='utf-8'));lines=[]
 def met(n,s):return next(x for x in e['comparison'] if x['model']==n and x['split']==s)
 def group(n,s,kind,g,op='F1'):return next(x for x in e['groups'] if x['model']==n and x['split']==s and x['kind']==kind and x['group']==g and x['op']==op)
 for n,b in [('A4a','A1'),('A4b','A3')]:
  a=met(n,'val');base=met(b,'val');at=met(n,'test');bt=met(b,'test')
  lines.append(f"**{n} 相对 {b}**：验证AUROC变化{a['auroc']-base['auroc']:+.4f}、AP变化{a['ap']-base['ap']:+.4f}、F1变化{a['f1']-base['f1']:+.4f}；探索性测试AUROC变化{at['auroc']-bt['auroc']:+.4f}、AP变化{at['ap']-bt['ap']:+.4f}、F1变化{at['f1']-bt['f1']:+.4f}。默认工作点测试FP {bt['fp']}→{at['fp']}、FN {bt['fn']}→{at['fn']}，召回{bt['recall']:.4f}→{at['recall']:.4f}。")
  for s in ['val','test']:
   ma=next(x for x in e['matched_recall'] if x['model']==n and x['split']==s and x['target']==str(154/199));mb=next(x for x in e['matched_recall'] if x['model']==b and x['split']==s and x['target']==str(154/199))
   lines.append(f"主验证召回要求77.39%下，{s}：FP {mb['fp']}→{ma['fp']}，实际召回{mb['recall']:.4f}→{ma['recall']:.4f}。"+('验证目标相同，可直接比较本点误报。' if s=='val' else '若实际测试召回不同，该FP差值不能单独视为等召回改善。'))
  for c in ['92','97','98']:
   ca=next(x for x in e['codes'] if x['model']==n and x['split']=='test' and x['code']==c);cb=next(x for x in e['codes'] if x['model']==b and x['split']=='test' and x['code']==c)
   ga=group(n,'test','code',c);gb=group(b,'test','code',c);va=next(x for x in e['codes'] if x['model']==n and x['split']=='val' and x['code']==c);vb=next(x for x in e['codes'] if x['model']==b and x['split']=='val' and x['code']==c)
   lines.append(f"代码{c}：验证分代码AUROC {vb['auroc']:.4f}→{va['auroc']:.4f}；测试 {cb['auroc']:.4f}→{ca['auroc']:.4f}；测试默认阈值漏检 {gb['errors']}→{ga['errors']} / {ga['n']}。排序能力和工作阈值效应应分开解释。")
  lines.append('测试默认阈值正常品相误报：'+'；'.join(f"{g} {group(b,'test','grade',g)['errors']}→{group(n,'test','grade',g)['errors']}" for g in ['AU','XF','MS'])+'。这组比较须与上文实际异常召回一并阅读。')
 for n in ['A4a','A4b']:
  for s in ['val','test']:
   d=json.loads((O/f'{n}_{s}_reason_diagnostic.json').read_text(encoding='utf-8'));cr=d['classification_report']
   lines.append(f"{n} {s} 原因头宏F1={cr['macro avg']['f1-score']:.4f}，全猜正常基线={d['majority_macro_f1']:.4f}；在三种异常之间强制选择时准确率={d['abnormal_only_conditional_accuracy']:.4f}，全猜92基线={d['abnormal_only_majority_baseline']:.4f}。这是原因标签相关性的诊断，不证明识别了物理成因。")
  b='A1' if n=='A4a' else 'A3';av=met(n,'val');bv=met(b,'val');gate=f['decision'][n]['paired_multiseed_worth_followup']
  lines.append(f"{n} 的验证多种子门槛判定：{'通过，建议后续使用配对种子确认' if gate else '未通过，本轮不扩展多种子'}。此决定在新测试推理前冻结。")
 va=met('A4a','val');vb=met('A4b','val');lines.append(f"两种监督组合直接比较：A4a/A4b验证AUROC={va['auroc']:.4f}/{vb['auroc']:.4f}，AP={va['ap']:.4f}/{vb['ap']:.4f}。{'A4b' if vb['auroc']>va['auroc'] else 'A4a'}的本次验证AUROC更高；是否值得采用还须看直接基线的增量、主召回点和全部分组，不以单个测试指标选方案。")
 with (O/'error_transitions.csv').open(encoding='utf-8-sig',newline='') as h:tr=list(csv.DictReader(h))
 lines.append('**组合有效性判断**：两组验证AUROC只差0.00008，不能据此排序。A4a相对A1的AUROC增量及部分召回点表现更值得保留为局部信号；A4b相对A3未体现稳定互补性。测试默认工作点A4b与A1召回恰同为71.86%，FP由112降为102，是局部正证据；但A4b相对A3的测试AUROC/AP均下降。主召回要求下A4a与A4b测试实际召回同为74.37%，FP分别108和114；这仅支持该冻结操作点A4a较好，不能宣称全面胜出。')
 lines.append('**召回约束的完整趋势**：A4a相对A1在验证70%、75%、80%、85%、90%要求下FP分别减少20、3、6、50、17枚，但主77.39%要求反增4枚。测试70%要求下FP93→73伴随召回69.85%→68.84%；85%下FP193→145伴随召回85.93%→79.90%，不能将这些下降全部视为等召回收益。A4b相对A3在80%要求下测试召回78.39%→78.89%接近，FP123→139；70%下召回70.35%→69.85%接近，FP83→90，均不支持一致减少误报。')
 lines.append('**原因类型信息与类别不均衡**：补充异常三类的宏F1／平衡准确率后，A4a验证为0.5939／0.5840，测试0.5490／0.5255；A4b验证0.5251／0.6076，测试0.4981／0.5404。三类全猜92的宏F1仅0.2638、平衡准确率1/3。两组都保留了原因相关信息；A4b强制三类测试预测把112枚92中的55枚分到98，说明它更偏少数类，不能因总体准确率低于多数基线就判定毫无区分作用。详见reason_anomaly_only.json。其信息没有稳定改善共享二分类排序，是本轮核心区别。')
 lines.append('**分数与训练长度限制**：A4a验证F1阈值约3.50e-4，主召回阈值约1.31e-6；A4b分别约0.0708和0.0567。分数不视为校准概率，跨模型绝对数值不代表同等置信度。A4a最佳21轮、实际31轮，A4b最佳4轮、实际15轮；统一预算规则下实际步数不同，单种子比较不能分离训练路径与辅助标签的全部影响。')
 lines.append('**可追踪的错例变化**（仅标签与分数证据，无视觉成因判定）：')
 for n,b in [('A4a','A1'),('A4b','A3')]:
  for typ in ['corrected','introduced','both_wrong']:
   cases=[x for x in tr if x['split']=='test' and x['model']==n and x['baseline']==b and x['code']=='92' and x['transition']==typ]
   lines.append(f"{n}相对{b}，92类{typ}共{len(cases)}枚，按编号前3例："+'；'.join(f"{x['id']}（分数{float(x['before_score']):.3g}→{float(x['after_score']):.3g}）" for x in sorted(cases,key=lambda x:x['id'])[:3])+'。')
 lines.append('**清洗为什么仍会漏检？** 本轮只能确认哪些带92标签的图像在冻结阈值下仍低分，不能仅据分数、品相或问题代码确定视觉原因。图像是否缺少可见证据、局部线索是否被缩放/池化削弱、成像与正常保存变化是否造成混淆，均需原图局部检查和专家盲审验证，不能写成已证明的钱币知识。')
 lines.append('**全局分类的局限**：当前每面全局池化后仅有768维，标签没有指定异常区域、正常设计参照或判断步骤。原因监督可能改善标签相关表征，也可能强化成像或保存状态的捷径；现有指标不能区分这两种机制。原因头有预测能力，不等于共享二分类头获得稳定收益，更不等于完成专家知识引导。')
 positive=any(f['decision'][n]['paired_multiseed_worth_followup'] for n in ['A4a','A4b'])
 lines.append('**明确结论**：'+('原因辅助监督达到预设验证门槛，存在值得配对多种子确认的局部增量信号，但单种子与探索性测试不足以证明稳定有效。' if positive else '本轮未达到预设验证增量门槛，不支持把异常原因辅助监督视为稳定有效的整体改进；局部指标收益仍可作为后续假设。')+'专家知识引导仍值得研究，因为本轮只检验标签监督，没有直接检验类别特定的正常参照和局部专家判断。后续建模优先级应放在同类别、相近正常品相的局部参照比较与专家证据核实，而不是继续堆叠全局分类头。'+('配对多种子确认可用于检验本轮信号，且应沿用冻结协议。' if positive else '先确认局部证据与参照是否能解释误报/漏检，再决定是否扩大训练。'))
 text='\n\n'.join(lines);(O/'interpretation.md').write_text(text,encoding='utf-8');p=R/'实验报告.md'
 if p.exists():
  doc=p.read_text(encoding='utf-8')
  if '<!-- INTERPRETATION -->' in doc:doc=doc.replace('<!-- INTERPRETATION -->',text)
  else:
   start=doc.index('## 结果解释与明确结论')+len('## 结果解释与明确结论');end=doc.index('![训练曲线',start);doc=doc[:start]+'\n\n'+text+'\n\n'+doc[end:]
  p.write_text(doc,encoding='utf-8')
if __name__=='__main__':main()
