from pathlib import Path
import json, html

ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'结果/review_manifest.json').read_text(encoding='utf8'))
stats=json.loads((ROOT/'结果/准备统计.json').read_text(encoding='utf8'))
qa=json.loads((ROOT/'结果/网页核验.json').read_text(encoding='utf8'))
sections=[
('本轮结论',[
'A6-1 集中可视化已完成，尚未形成经专家确认的局部监督信号，也没有启动 A6 正式训练。现有照片中能看见细线、表面颗粒、轮廓强弱和色泽差别；这些可见现象是否足以可靠区分正常保存变化与清洗等损伤，仍需人工结合参照逐币判断。不能把“能看到纹理”直接等同于“能确认损伤原因”。',
'承接 A0+A1 全局基线；A2 的真实类别条件不等于自动类别预测；A3/A3b 品相辅助监督存在误报与漏检取舍；A4 原因标签有信息但未形成一致二分类增益；A5 同类别正常参照有互补性；A5.1 对齐、高分辨率和裁剪边界问题未支持继续堆叠局部距离。本轮只准备有限图像证据核验，不复跑这些旧实验。']),
('固定选样与集合边界',[
'全部 30 枚目标来自验证集：正常 AU/XF 误报 10 枚（各 5 枚）、92 清洗漏检 10 枚、92 清洗正确检出 4 枚、正常 AU/XF 正确识别 4 枚（各 2 枚）、其他异常 2 枚（97、98 各 1 枚）。两面、证书号始终属于同一枚币。没有训练集目标补样。',
'固定使用 A1/融合对照/A0_A1集成/frozen_ensemble_config.json 的阈值 0.08103002402931452；分数 ≥ 阈值判异常，分数公式为 0.2×A0 + 0.8×A1。此值也与 A5 保存的 A0+A1 maxF1 工作点相同；没有改用 A5 的等召回工作点或重新寻优阈值。直接使用历史 val_predictions.csv 分数。',
'对应候选数为 69、18、94、188、59 枚。每个选样层按既有分数排序分成等数量区间，在区间内优先类别多样性，再用固定证书哈希打破并列；网页顺序另按固定哈希混排。所有 18 枚清洗漏检中取 10 枚，不是只挑分数最低的例子，也未观察图片后再换样。这是目的性错误分析样本，不能估计总体患病率或总体准确率。',
'选样、匹配、图片读取仅涉及训练和验证来源。未打开历史测试清单、预测或图片；历史报告只用于继承研究结论，不用于选样。所有新增文件均在 A6，A0–A5.1 的文件、权重和缓存未写入、覆盖或删除。']),
('正常参照与图像',[
f'每币 2 枚正常参照，60 个目标—参照配对，共 {stats["references_unique"]} 枚不同训练正常币；172 张双面图片原样复制，共约 {stats["image_bytes"]/1024**2:.1f} MiB。复制图像均为 1024×1024，字节哈希与来源一致，没有为了网页重新压缩、缩小或锐化。缩略特征只参与候选排序，不替代展示图片。',
'以 A2 已确认的类别为条件，先找同类别训练正常币，再优先版别字符串一致。使用已存在的 A5 双面特征均值及余弦相似度，每个目标最多保留 6 枚图像短名单，再以 0.7×特征距离 + 0.3×低频 RGB 外观距离排序取 2 枚。只复用既有特征，没有新模型训练、品相预测或局部距离调参。缓存未覆盖的候选排在有缓存者之后；因此这是便于审核的短名单，不是全库最优参照保证。',
'匹配函数只接收目标编号、类别、版别和双面图片路径，不接收验证币真实品相、异常代码或真值标签。正常训练评级仅随参照保存以供显示，未按验证真值挑同品相参照。60 次参照显示中的评级组为 XF 20、AU 17、MS 21、VF及以下 2；这些是配对次数，不是去重币数。相似成像不保证相同磨损程度。',
'本包 30 枚均无需类别回退。17 号目标 45979066 的参照 2（88034888）虽同类别，版别字符串不同，网页已黄色提示；请核对是否适合。其余 59 对版别字符串一致，字符串一致也不代表重新完成版别鉴定。若后续不足，代码只允许明确的上级类别回退，不静默全库混配。']),
('有限目视发现：只描述现象',[
'以下是助手对 3 组代表性对照的初步浏览，不是专家盲审，也不是完成了全部 30 枚的损伤认定；人工观察模板仍全部留空。',
'19 号 38540371（正常 XF 误报）与 47836490：两者正面底板与肖像附近都可见细线，目标肖像轮廓和发丝细节更弱，参照反光及黄灰色分布不同。目标背面也存在细线。仅凭“有细线”不足以区分异常；应核对线条分布、纹饰高低点关系和同类正常范围。',
'18 号 45842252（92 漏检）与 45176160：两者均有棕灰色表面和细微纹理，目标肖像细节更弱、参照发丝更清楚且色泽更偏红褐。目标背面文字与嘉禾周围纹理可见，但目前未确定哪一区域构成清洗证据；磨损、表面状态和拍摄条件的作用不能据这对图像拆开。',
'11 号 46083392（92 检出）与 40787081：目标底板、脸部附近可见多方向线状变化；正常参照的额部和右侧底板也有可见线条。高异常分数不能替代清洗成因判断，参照明显更亮也可能干扰对比。这组适合人工核对“具体纹理形态是否可分”，而非按有无线条直接分类。',
'这些发现说明图片保留了可供局部观察的信息，同时正常与异常标签之间存在外观重叠。当前没有充分证据断言局部判别一定可学，也不能断言现有照片完全不可辨识。关键缺口是可靠的人为判断及可定位程度。']),
('怎样记录并做出下一步决定',[
'双击“人工观察.html”直接离线打开。先隐藏真实标签与热图；分数也可关闭。对正面／背面分别比较底板细线、高点、文字和纹饰邻域，再看正常参照是否有相似纹理。滚轮缩放、拖动平移；可展开双图或取消联动分别定位。联动没有进行几何配准，不应把同屏坐标视为严格解剖对应。',
'每币标记四种判断之一，并选填部位、备注、能否定位。“疑似异常证据”仍不是专家确认的训练标签；“参照不合适”应说明哪一枚参照以及原因。导出时记录当前参照、已看过的参照及是否曾显示标签／分数／热图。一个浏览器的本地存储不是共享文件，请将导出文件放入“人工观察结果”目录；JSON 可导入恢复。',
'建议在核对后按五个选样组分别统计：可见疑似证据、正常变化、无法判断、参照不合适的枚数，并查看“可见证据”中有多少能够可靠定位。对关键例可再独立复核并记录分歧；不以强制二选一替代无法判断，也不从此小包计算模型增益。',
'A5b 热图来自 A5 的旧 24×24 网格缓存，色标沿用旧可视化的距离 99% 分位上限。该热图基于原有 32 枚训练参照，不是当前两枚展示参照的差异图；模型热区不能自动转为真实损伤框。']),
('下一阶段训练建议：等人工核验后选择',[
'情况一——局部证据能被可靠确认：另在训练集中采集同类别、对应面、相近成像的经确认正常／异常局部区域；保存证书、正反面、原图坐标、原因与确认者。以轻量局部编码器或冻结特征上的小判别头做起点，用分类损失配合监督对比损失，重点组织“正常磨损／接触细纹”与“确认异常纹理”的困难对照。正常与异常类别内也要覆盖不同照明及磨损，避免只学亮度。仅给被确认的区域局部标签，未审区域保持未知；这比直接给整币所有切块贴 92 更符合本轮目标。',
'情况二——整币异常可判断但无法可靠定位：优先把一枚币的双面局部块组织成一个 MIL 包，以训练集整币标签监督轻量注意力聚合器和局部表征。包内切块不继承统一的清洗标签，注意力仅代表对模型判别的贡献，也不自动等于损伤位置。以整币为划分和统计单位，控制边缘背景、裁剪接缝及颜色线索；正常 AU/XF 应作为有针对性的困难负例。',
'若照片本身普遍无法判断或参照不合适，应先解决参照／可见性问题；不能因为无法定位就直接假定 MIL 可以解决。局部监督与 MIL 的选择依据是确认程度，不是 A5 热区强度。',
'后续监督仅来自训练集。此包 30 枚及其人工观察结果继续作为验证探索／评价信息，不能混入训练区域或对比样本；训练参照在此页的验证比较备注也不能直接当作其局部监督。两面及所有切块保持同证书归属。保留 A0+A1 固定分数作为比较基线，观察 AU/XF 误报与 92 漏检的共同变化；反复使用的验证集只支持探索性结论。']),
('交付、核验与复现',[
'主入口：人工观察.html；报告：分析报告.html／分析报告.txt；图片：图片/；辅助热图：辅助热图/；审阅清单与参照映射：结果/样本清单.csv、结果/参照匹配.csv、结果/review_manifest.json；来源与复制哈希：结果/图像来源.json。清单和报告含真值标签，建议盲看图片后再阅读。',
'结果/人工观察_空白模板.json 与 .csv 保留 30 条未观察记录。正式人工结果由网页导出，不预填专家判断。结果/网页核验.json 记录本轮针对交付物的检查：所有双面图片离线加载、默认隐藏标签／热图、缩放拖动联动与解联动、原像素比例、展开视图、记录持久化、CSV/JSON 导出及 JSON 导入；浏览器无脚本错误。并已目视检查桌面网页截图。',
'代码/prepare.py 准备样本、复制原图并内嵌数据生成网页；代码/template.html 是界面源码；代码/verify_page.py 只检查 A6 交付物；代码/write_report.py 生成报告。复现使用 Python（numpy、Pillow；网页核验另需 Playwright 与已安装 Edge），按 prepare → verify_page → write_report 顺序运行。查看网页无需 Python、服务器或网络。',
'本轮停止于图片审核准备。未启动正式 A6 训练、未新增随机种子实验、未重训主干、未建立专家知识库。待用户核对实际图片与记录后再确定监督机制。'])]

title='A6-1 正常品相与异常损伤：小规模视觉证据核验'
body='<p class="notice">报告含真实评级和误报／漏检分组。建议先完成无标签观察，再阅读本报告。</p><p>2026-11-14 · '+data['dataset_id']+'</p><p><a href="人工观察.html">返回人工观察网页</a></p>'
for heading,paragraphs in sections:
    body+='<h2>'+html.escape(heading)+'</h2>'+''.join('<p>'+html.escape(p)+'</p>' for p in paragraphs)
page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><style>body{max-width:920px;margin:36px auto;padding:0 24px;color:#213b3b;background:#f7f9f7;font:16px/1.85 "Microsoft YaHei",sans-serif}h1{font-size:28px}h2{font-size:21px;margin-top:32px;border-top:1px solid #ccd9d4;padding-top:20px}a{color:#126f65}.notice{background:#fff1d7;padding:12px 16px;border-radius:7px}@media print{body{background:white;font-size:11pt}}</style><h1>'+title+'</h1>'+body+'</html>'
(ROOT/'分析报告.html').write_text(page,encoding='utf8')
(ROOT/'分析报告.txt').write_text(title+'\n2026-11-14\n\n'+'\n\n'.join(h+'\n'+'\n\n'.join(ps) for h,ps in sections),encoding='utf-8-sig')
(ROOT/'人工观察结果').mkdir(exist_ok=True)
(ROOT/'人工观察结果/保存说明.txt').write_text('请将网页导出的 CSV 或 JSON 保存到此处。\n当前没有专家已完成的人工观察结果；空白模板位于 ../结果/。\n验证集观察仅用于探索、评价和方案分析，禁止直接混入训练监督。\n',encoding='utf-8-sig')
(ROOT/'打开说明.txt').write_text('双击“人工观察.html”即可离线使用，不需要服务器。\n图片目录和辅助热图目录必须与网页一起保留。\n默认隐藏真实标签及热图，分数可关闭。滚轮放大、拖动移动，可展开对应双面。\n逐币选择判断并选填部位与备注；右上角导出 CSV/JSON，保存到“人工观察结果”目录。\n浏览器暂存不是共享磁盘文件，请定期导出；JSON 可导入恢复。\n本轮 30 枚目标全来自验证集，每币 2 枚正常训练参照。没有启动训练。\n报告含真值标签，建议盲看图片后再读。\n',encoding='utf-8-sig')
protocol={'stage':'A6-1','status':'ready_for_human_review_no_training','date':'2026-11-14','dataset_id':data['dataset_id'],
 'threshold':data['threshold'],'threshold_source':'A1/融合对照/A0_A1集成/frozen_ensemble_config.json',
 'selection':'fixed score-quantile strata + category diversity + certificate hash tie-break; blind mixed order',
 'target_split':'val','reference_split':'train','target_count':30,'reference_count_per_target':2,
 'matching_query_fields':['钱币编号','category_name','category_reason','版别','正面路径','背面路径'],
 'target_true_grade_used_for_matching':False,'known_category_condition':True,'automatic_category_prediction':False,
 'training_started':False,'historical_test_manifests_predictions_images_read':False,
 'writes_scope':'A6 only','feature_cache':'A5/特征缓存/<train_normal_or_selected_val_id>.npy',
 'heatmap_source':'A5/结果/local_maps.npz A5b only','heatmap_is_ground_truth':False,
 'future_supervision':'train only; validation review cannot be mixed into training',
 'human_results_status':'blank; awaiting user review'}
(ROOT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
print('Report and usage notes saved.')
