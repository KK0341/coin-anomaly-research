from pathlib import Path
import hashlib,json
R=Path('E:/异常检测/算法训练'); O=R/'A10_预训练骨干对照实验'
paths=[R/'A8_高分辨率实验/代码/train.py',R/'A8_高分辨率实验/模型/best.pt',R/'A8_高分辨率实验/模型/config.json',R/'A8_高分辨率实验/结果/val_predictions.csv',R/'A1/结果/val_predictions.csv',Path('E:/异常检测/数据划分/训练集.csv'),Path('E:/异常检测/数据划分/验证集.csv')]
a={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
(O/'来源/protected_input_hashes.json').write_text(json.dumps(a,ensure_ascii=False,indent=2),encoding='utf8')
(O/'实验报告.md').write_text('''# A10：预训练视觉骨干对照实验（训练进行中）

DINOv3官方权重访问返回HTTP401受限，已按预定规则切换DINOv2 ViT-S/14无Registers版本。
已严格加载官方全部175个权重键并逐张量核验；单面384维、双面768维，骨干全部可训练。前向、反向及早/末层真实参数更新检查通过。
正式训练：504×504、batch8、FP16、原A8优化器和选模/早停规则；只在一张动态选择的空闲GPU运行。
当前尚无最终结论，不能声称超过A8。训练完成后自动导出755枚验证预测、指标、错误对照并更新本报告。
详见结果/train.log、结果/history.csv及模型/config.json。无测试集评估、无旧模型重训。
''',encoding='utf8')
print('source snapshot saved')
