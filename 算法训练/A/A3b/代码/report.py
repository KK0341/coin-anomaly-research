"""Generate Chinese report from frozen outputs; never choose settings."""
import json,csv,hashlib,collections
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'结果';M=R/'模型'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(rows,cols):
 def fmt(v):return f'{v:.4f}' if isinstance(v,float) else str(v)
 return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(fmt(r.get(c,'')) for c in cols)+' |' for r in rows])
e=js(O/'evaluation.json');f=js(M/'frozen_evaluation.json');done=js(O/'TRAIN_COMPLETE.json');cfg=js(M/'config.json');stats=js(O/'grade_summary.json');dist=read(O/'added_abnormal_distribution.csv');trans=read(O/'error_transitions.csv')
test={r['model']:r for r in e['comparison'] if r['split']=='test'};val={r['model']:r for r in e['comparison'] if r['split']=='val'}
primary=[r for r in e['matched_recall'] if r['split']=='test' and r['target']==f['primary_recall_target']];mp={r['model']:r for r in primary}
better=test['A3b']['auroc']>test['A3']['auroc'] and test['A3b']['recall']>=test['A3']['recall'] and mp['A3b']['fp']<mp['A3']['fp'] and mp['A3b']['recall']>=mp['A3']['recall']-.02
conclusion=('单次实验提供有限支持，但不能据此确认稳定收益。' if better else '本次单种子结果不足以支持“共享品相监督更有利于异常检测”的整体假设；局部改善与退步应分别判断。')
summary=[]
for split,d in stats.items():summary.append({'split':split,**d})
group_counts=collections.Counter();code_counts=collections.Counter()
for r in dist:
 if r['split']=='train':group_counts[r['grade_group']]+=int(r['n']);code_counts[r['abnormal_code']]+=int(r['n'])
tr=[]
for baseline in ['A1','A3']:
 for label in ['0','1']:
  cnt=collections.Counter(r['transition'] for r in trans if r['split']=='test' and r['baseline']==baseline and r['二分类标签']==label)
  tr.append({'baseline':baseline,'类别':'正常误报' if label=='0' else '异常漏检',**{k:cnt[k] for k in ['corrected','introduced','both_wrong','both_right']}})
parts=['# A3b：正常与异常共享品相监督实验报告',
'## 结论与研究边界',conclusion,
f"相对 A3，测试 AUROC 变化 {test['A3b']['auroc']-test['A3']['auroc']:+.4f}，AP 变化 {test['A3b']['ap']-test['A3']['ap']:+.4f}；验证 F1 阈值下 Recall 变化 {test['A3b']['recall']-test['A3']['recall']:+.4f}，FP 从 {test['A3']['fp']} 变为 {test['A3b']['fp']}，FN 从 {test['A3']['fn']} 变为 {test['A3b']['fn']}。A3 相对 A1 新增的 {e['A3_new_FN_count']} 枚漏检中，A3b 恢复 {e['A3_new_FN_recovered']} 枚。",
'这是探索性冻结评估：本轮假设已由历史测试错例启发，因此不能宣称是完全未触碰测试集的确认性研究。本轮模型与阈值只用验证集确定，冻结后仅进行一次测试集图像推理；错例仅在此后分析，没有反馈调参。',
'## 数据与唯一实验变量',
'Excel 的“钱币详细信息”表只读提取证书号和原始评级，并与 A3 清单逐枚核对。正常币可靠标签逐字段原样继承。异常币仅接纳明确 Genuine - [保存状态] Details (...) 描述，VF 及以下合并，XF、AU 各自成组。UNC Details 及无数字 MS Details 均排除，不映射到数字 MS60–70；异常代码仅用于事后分组，绝不当作品相。无法可靠标注的样本仍进入 BCE。测试清单的评级只用于描述性分析，不用于模型选择。',
table(summary,['split','total','abnormal','normal_supervised','new_abnormal_supervised']),
'训练集新增分组：'+json.dumps(dict(group_counts),ensure_ascii=False)+'；按异常代码：'+json.dumps(dict(code_counts),ensure_ascii=False)+'。所有划分的品相×异常代码交叉统计见 `结果/added_abnormal_distribution.csv`，逐枚依据见 `结果/added_abnormal_grade_rows.csv`。',
'`代码/train.py` 与 A3 唯一源码差异为辅助损失掩码从 `(y==0)&(grade>=0)` 改为 `(grade>=0)`，diff 已保存。仍为 L = BCE + 0.2 × CE；CE 在每个 batch 的可靠标注样本上取均值。新增监督改变了辅助学习样本分布，不能将收益直接解释为已学到严格解耦的表示。',
'## 训练与复现',
f"seed=42；共享 ConvNeXt-Tiny，双面特征拼接 1536 维，异常 MLP 1536→256→1，品相 Linear 1536→4。ImageNet 原始权重初始化。batch=8 枚/16 图、384×384、AdamW、主干学习率 2e-5、头部 1e-4、weight decay=0.01、梯度裁剪=5、AMP、独立 ±10° 旋转。原始划分、配对、缓存、加载器、调度及早停规则均沿用 A3。最多50轮、最少15轮、早停patience=10。实际 {done['epochs']} 轮，最佳第 {done['best_epoch']} 轮，训练耗时 {done['training_seconds']/60:.2f} 分钟。",
'模型选择只看验证 AUROC，阈值最大化验证 F1，并列选最高阈值。无融合、新模块、复核网络或超参数搜索。推理 forward 仅接收图像张量；真实类别、品相及异常代码不输入模型。首轮仅完成 seed=42，未据测试结果追加种子。即使验证改善，稳定性仍需未来配对多种子研究，不能从本轮推断。',
'复现顺序：使用 `模型/config.json` 中的 python 解释器运行 `代码/prepare.py`、`代码/train.py --mode train`、`代码/evaluate.py freeze`、`代码/evaluate.py test`、`代码/report.py`。应在新的独立目录复现实验；当前目录有防重复准备和测试推理保护。训练中断可使用 last.pt 恢复。',
'## 验证集与冻结测试结果',table(e['comparison'],['split','model','threshold','auroc','ap','f1','precision','recall','fp','fn','tp','tn']),
'## 相近召回要求下的 FP 比较',
f"预设目标为 0.70、0.75、0.80、0.85、0.90，主目标为 A1 原验证 F1 阈值下 Recall={f['primary_recall_target']:.4f}。每个模型独立选择满足目标的最高验证阈值，再原样应用测试集。以下报告实际召回，测试召回可能不相等，不能把明显较低召回下的 FP 减少当作有效改善，也没有在测试集重选阈值。",
table(e['matched_recall'],['split','model','target','threshold','recall','fp','fn','precision']),
f"主目标下：A1 测试 Recall={mp['A1']['recall']:.4f}/FP={mp['A1']['fp']}；A3 为 {mp['A3']['recall']:.4f}/{mp['A3']['fp']}；A3b 为 {mp['A3b']['recall']:.4f}/{mp['A3b']['fp']}。这一比较需要同时看召回偏差及所有预设操作点，不能只看默认阈值 FP。",
'## 分组错误与错误转移',
table([r for r in e['groups'] if r['split']=='test' and (r['operating_point']=='validation_F1' or r['operating_point']==f['primary_recall_target']) and (r['kind']=='normal_grade' or r['group'] in ['92','97','98'])],['model','operating_point','kind','group','n','errors','error_rate']),
'normal_grade 的 errors 为正常 FP，abnormal_code 的 errors 为异常 FN；分母 n 均为该组测试样本数。完整阈值网格分组结果见 `结果/group_errors.csv`。',
table(tr,['baseline','类别','corrected','introduced','both_wrong','both_right']),
f"已知 A3 新增漏检集合由冻结后的 A1 正确异常、A3 错误异常重新计算，数量为 {e['A3_new_FN_count']}，与用户提供的16枚 {'一致' if e['A3_new_FN_count']==16 else '不一致，应以保存的逐枚清单核查'}。恢复 {e['A3_new_FN_recovered']} 枚，仍漏检 {e['A3_new_FN_count']-e['A3_new_FN_recovered']} 枚。逐枚分数、评级与代码见 `结果/A3_added_FN16.csv`；全部错误转移见 `结果/error_transitions.csv`。该集合没有被训练、早停或阈值选择使用。",
'## 保存与核验',
'独立目录保存核心代码、配置、原始评级依据、训练日志、历史、best.pt、last.pt、验证及测试预测、冻结配置与比较表。历史 A0/A1/A2/A3 文件逐文件 SHA256 在准备后及报告时比对；缓存只读复用，无清理或覆盖。完整检查见 `结果/verification.json`。']
parts.insert(3,'本轮未恢复 A3 损失的异常召回；正常误报改善基本保留。相对 A3，AU 正常币误报保持23枚，XF 从21降至20枚。异常代码92漏检保持38枚，97从9增至12枚，98从10降至9枚。相对 A3 共纠正6枚异常漏检、引入8枚新漏检；正常币纠正7枚误报、引入5枚新误报。已知16枚中恢复5枚不能代替全体异常样本的净效果。')
parts.insert(4,'召回约束下没有一致优势：验证目标0.70对应测试 A1/A3/A3b Recall=0.6985/0.7035/0.6985，FP=93/83/73，支持较低召回操作点的局部改善；目标0.85对应 Recall=0.8593/0.8141/0.8090，FP=193/142/159，A3b 相对 A3 在略低召回下反而多17枚 FP。主目标下 A3b 比 A3 少19枚 FP，但召回也低4.02个百分点，不能归为等召回收益；相对 A1，A3b 此点召回高1.01个百分点且 FP 少15枚。目标0.80下 A1 与 A3b 的测试召回同为0.7487，FP由137降至123，但 A3 在更高召回0.7839下同为123枚 FP。因此只支持部分操作点的改善，不支持全面优于正常币单独品相监督。')
parts.insert(5,'历史继承关系：A0 为双面共享骨干、单面分数 max 融合的基线；A1 改为双面池化特征拼接与 MLP；A2 在 A1 上增加真实币种类别 Embedding，存在额外类别输入；A3 在 A1 上使用仅正常币监督的品相头。A3b 严格以 A3 为控制组，不引入 A2 的类别输入，也不加入既有融合方案。所有历史成果均保留。验证集 AUROC/AP/F1 有改善，但相同验证召回网格并非处处改善，因此本轮完成首个单种子实验后不追加配对种子；这是预算范围决定，不能把未验证稳定性当作稳定收益。')
historical=js(O/'historical_hashes.json');changed=[]
for path,h in historical.items():
 p=Path(path)
 with p.open('rb') as stream:now=hashlib.file_digest(stream,'sha256').hexdigest()
 if now!=h:changed.append(path)
assert not changed,changed
assert e['A3_new_FN_count']==16
checks={'historical_files_checked':len(historical),'historical_files_changed':changed,'known_A3_FN_count':e['A3_new_FN_count'],'test_prediction_rows':len(read(O/'test_predictions.csv')),'single_test_inference_marker':js(O/'TEST_INFERENCE_STARTED.json'),'source_diff_only_loss_mask':(R/'代码/train.py').read_text(encoding='utf-8')==(R.parent/'A3/代码/train.py').read_text(encoding='utf-8').replace('mask=(y==0)&(grade>=0)','mask=(grade>=0)')}
checks['post_freeze_reporting_only']='report.py updated to explain frozen results; training, evaluation, manifests and thresholds unchanged'
for name in ['train.py','prepare.py','evaluate.py']:
 with (R/'代码'/name).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==f['code_sha256'][name]
assert checks['source_diff_only_loss_mask'] and checks['test_prediction_rows']==755
(O/'verification.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
(R/'实验报告.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8');(O/'COMPLETE.json').write_text(json.dumps({'status':'complete','epochs':done['epochs'],'best_epoch':done['best_epoch'],'test_status':'exploratory frozen evaluation'},ensure_ascii=False,indent=2),encoding='utf-8')
print(conclusion,flush=True)
