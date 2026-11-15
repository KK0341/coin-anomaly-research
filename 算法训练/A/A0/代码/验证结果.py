"""Audit saved outputs only; never runs inference or changes model selection."""
import csv, hashlib, json
from collections import Counter
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve
from train import ROOT, OUT, MODEL, SPLIT, FILES, metrics, save_json

def read_csv(path):
    with open(path,encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def report_intro(config,result,history):
    n=len(history); avg=result['training_seconds']/n; first=history[0]
    training=sum(h['seconds'] for h in history)
    throughput=6033*n/training
    wait=sum(h['data_wait_seconds'] for h in history)/training
    bench=json.loads((OUT/'batch_benchmark.json').read_text(encoding='utf-8'))
    text=f'''# A0 袁世凯银币双面监督二分类基线

已完成 **{n} 轮**，连续 10 轮验证 AUROC 未刷新后早停。最佳模型为 **Epoch {result['best_epoch']}**，验证 AUROC **{max(h['val_auroc'] for h in history):.4f}**。测试集仅在模型和阈值确定后进行一次推理评估。

## 数据、环境与配置

- 固定 Dataset V1：train 6033（正常 4444 / 异常 1589），val 755（556 / 199），test 755（556 / 199）。双面保持整币配对；未修改原图、Excel 或划分。CSV 哈希核验通过。
- 复用已有“币缘改进环境”：Python 3.13.9、PyTorch {config['torch']}、torchvision {config['torchvision']}、CUDA {config['cuda']}。GPU：{config['gpu']}，总显存 {config['total_MiB']:.0f} MiB。未修改该环境依赖。
- ImageNet 预训练 ConvNeXt-Tiny，全参数微调；正反面共享同一 Backbone 和分类头。整币 logit = max(正面 logit, 背面 logit)，BCEWithLogitsLoss，正常 0 / 异常 1。
- 输入 384×384；训练仅小角度旋转 ±10°，保留完整币面，不使用裁剪、模糊、颜色强扰动或擦除。ImageNet 标准化。
- Batch **{config['batch_coins']} 枚 / {config['images_per_batch']} 张图像**，梯度累积 {config['accumulation']}，有效 Batch {config['effective_batch_coins']} 枚。
- AdamW，初始 Backbone / 分类头学习率 2e-5 / 1e-4，weight decay 0.01，梯度范数裁剪 5，AMP float16。
- 按用户中途确认的配置：最少 15、最多 50 轮，Early Stopping Patience=10，仅以验证 AUROC 选择最佳权重。前 5 轮保持初始学习率，第 6 轮起启用 ReduceLROnPlateau：mode=max、factor=0.5、patience=4、绝对提升阈值 1e-4、最小学习率 2e-7 / 1e-6。学习率下降不重置早停计数。
- 第 14 轮后将下一轮学习率降为 1e-5 / 5e-5；第 19 轮后再次下降，但随即满足早停条件，未进行第 20 轮。
- 基础随机种子 42；各轮使用 42+epoch 重置随机源，支持按完整轮次恢复。完整配置见 `模型/config.json`，环境版本见 `代码/environment.txt`。

## 真实训练测速与耗时

三档均读取真实双面训练图片，先预热 3 步，再测量 12 步；包含前向、反向、AdamW 更新，并进行 CUDA 同步计时。测速模型随后丢弃，正式训练重新加载 ImageNet 权重。

| Batch 枚 / 图 | 峰值分配 MiB | 峰值保留 MiB | 枚/秒 |
|---|---|---|---|
'''
    for b in bench:
        if b['ok']: text+=f"| {b['batch']} / {b['images']} | {b['allocated_MiB']:.0f} | {b['reserved_MiB']:.0f} | {b['coins_per_second']:.2f} |\n"
    text+=f'''
Batch 16 可运行，但未提升吞吐；选择 Batch 8 保留显存余量。

- 首轮训练 {first['seconds']:.1f} 秒，含验证、保存共 {first['epoch_total_seconds']:.1f} 秒，训练速度 {first['coins_per_second']:.2f} 枚/秒。
- 按首轮实测外推 10 / 20 / 30 轮：{first['epoch_total_seconds']*10/60:.1f} / {first['epoch_total_seconds']*20/60:.1f} / {first['epoch_total_seconds']*30/60:.1f} 分钟。
- 实际完整轮次平均 **{avg:.1f} 秒（{avg/60:.2f} 分钟）/轮**；完整轮次累计 **{result['training_seconds']/3600:.2f} 小时**。按该平均速度外推 50 轮约 {avg*50/3600:.2f} 小时。
- 从首轮开始到最终评估的总墙钟耗时约 **{result['wall_elapsed_seconds']/3600:.2f} 小时**，包含配置切换、一次未完成轮次及恢复。起点由首轮历史文件创建时间减实测首轮时长反推，存在很小的文件写入计时误差。
- 完整轮次整体训练吞吐 **{throughput:.2f} 枚/秒**，数据队列等待占训练时间 **{wait:.1%}**。E 盘为 USB 机械硬盘，数据读取管线是主要瓶颈。
- 正式训练峰值分配 / 保留显存 **{result['peak_allocated_MiB']:.0f} / {result['peak_reserved_MiB']:.0f} MiB**（PyTorch allocator；不等同于含驱动上下文的整卡占用）。

第 7 轮首次尝试在 AMP 非有限梯度保护处停止，从第 6 轮检查点恢复。已修复为缩小 scale、恢复 CUDA 随机状态并重试同一批次；不更新坏梯度、不跳过样本。真实第 751 批次重试成功。完整轮次共 {result['successful_epoch_amp_retries']} 次自动 AMP 重试，未发生 OOM。详见 `结果/amp_recovery.json`。

## 最终测试

阈值 **{result['threshold']:.9f}**，仅由最佳模型在验证集上最大化 F1 确定，并列选最高阈值；未使用测试集调参。曲线中的每轮验证 F1 使用固定阈值 0.5。

| 指标 | 测试结果 |
|---|---|
| Accuracy | {result['accuracy']:.4%} |
| Precision | {result['precision']:.4%} |
| Recall | {result['recall']:.4%} |
| F1 | {result['f1']:.4f} |
| AUROC | {result['auroc']:.4f} |
| Average Precision | {result['ap']:.4f} |

混淆矩阵按真实标签为行、预测标签为列：TN={result['confusion_matrix'][0][0]}、FP={result['confusion_matrix'][0][1]}、FN={result['confusion_matrix'][1][0]}、TP={result['confusion_matrix'][1][1]}。

![混淆矩阵]({(OUT/'confusion_matrix.png').as_posix()})

![训练 Loss 与验证 AUROC/F1]({(OUT/'training_curves.png').as_posix()})

## 文件与复现

- `模型/best.pt` 为最佳权重；`模型/last.pt` 包含优化器、AMP scaler 和调度器状态；`模型/threshold.json` 固定验证阈值。
- `结果/test_predictions.csv`、`val_predictions.csv` 保存编号、标签、整币及正反面各自的 logit 和分数。单面分数为弱监督证据，未单独校准。
- `结果/history.csv` 保存各轮指标、速度、显存、实际及下一轮学习率；`typical_errors.csv` 保存 30 个高置信度错误编号、标签和分数。
- 使用 `模型/config.json` 中的 Python 运行 `代码/train.py --mode train` 可从检查点恢复。使用 `代码/验证结果.py` 可核验已有结果，不重新进行测试推理。

未使用的 A0 `.venv` 是用户纠正前创建的目录，安装已停止；自动审批策略拒绝清理操作，因此该目录仍保留。实际训练全程复用 D 盘已有环境。
'''
    return text

def main():
    config=json.loads((MODEL/'config.json').read_text(encoding='utf-8'))
    result=json.loads((OUT/'test_metrics.json').read_text(encoding='utf-8'))
    history=json.loads((OUT/'history.json').read_text(encoding='utf-8'))
    assert [h['epoch'] for h in history]==list(range(1,len(history)+1))
    assert config['min_epochs']<=len(history)<=config['max_epochs']
    selected=int(np.argmax([h['val_auroc'] for h in history]))+1
    assert selected==result['best_epoch']
    if len(history)<config['max_epochs']: assert len(history)-selected>=config['patience']
    for split,file in FILES.items():
        assert hashlib.sha256((SPLIT/file).read_bytes()).hexdigest()==config['split_sha256'][split]
    threshold=result['threshold']; tables={}
    fusion_error=0.
    for split in ['val','test']:
        records=read_csv(OUT/f'{split}_predictions.csv'); tables[split]=records
        assert len(records)==755 and len({r['钱币编号'] for r in records})==755
        for r in records:
            assert int(r['true_label'])==int(r['二分类标签'])
            assert float(r['threshold'])==threshold
            assert int(r['pred_label'])==int(float(r['coin_score'])>=threshold)
            error=abs(float(r['coin_logit'])-max(float(r['front_logit']),float(r['back_logit'])))
            fusion_error=max(fusion_error,error); assert error<1e-7
            assert abs(float(r['coin_score'])-max(float(r['front_score']),float(r['back_score'])))<1e-7
            for prefix in ['coin','front','back']:
                assert abs(float(r[prefix+'_score'])-1/(1+np.exp(-float(r[prefix+'_logit']))))<2e-7
    val=tables['val']; vy=np.array([int(r['true_label']) for r in val]); vp=np.array([float(r['coin_score']) for r in val])
    precision,recall,thresholds=precision_recall_curve(vy,vp)
    f1=2*precision[:-1]*recall[:-1]/(precision[:-1]+recall[:-1]+1e-12)
    assert threshold==float(thresholds[np.flatnonzero(f1==f1.max())[-1]])
    test=tables['test']; y=np.array([int(r['true_label']) for r in test]); p=np.array([float(r['coin_score']) for r in test])
    recomputed=metrics(y,p,threshold)
    for k,value in recomputed.items(): assert np.allclose(value,result[k],rtol=0,atol=1e-12),k
    assert Counter(y)=={0:556,1:199}
    best=torch.load(MODEL/'best.pt',map_location='cpu',weights_only=False)
    assert all(torch.isfinite(t).all() for t in best['model'].values())
    initial=torch.load(MODEL/'convnext_tiny-983f1562.pth',map_location='cpu',weights_only=True)
    assert best['epoch']==selected and best['val_auroc']==history[selected-1]['val_auroc']
    assert tuple(best['model']['net.classifier.2.weight'].shape)==(1,768)
    keys=[k for k in initial if k.startswith('features.')]
    changed=sum(not torch.equal(initial[k],best['model']['net.'+k]) for k in keys)
    assert not torch.equal(initial['features.0.0.weight'],best['model']['net.features.0.0.weight'])
    assert changed>0
    checks={'status':'PASS','test_coins':len(test),'val_coins':len(val),'threshold_from_val_only':threshold,
            'best_epoch_matches_val_auroc_max':selected,'max_logit_fusion_error':fusion_error,
            'metrics_match_saved_predictions':True,'original_split_hashes_unchanged':True,
            'changed_backbone_tensors':changed,'backbone_tensors_compared':len(keys)}
    save_json(OUT/'结果核验.json',checks)
    text='\n\n## 错误分析摘要\n\n| 类型 | 错误方向 | 错误数 / 该类总数 | 错误率 |\n|---|---|---|---|\n'
    for g in result['groups']:
        rate=f"{g['error_rate']:.1%}" if g['error_rate'] is not None else '无样本'
        text+=f"| {g['type']} | {'误报' if g['error_kind']=='FP' else '漏检'} | {g['errors']} / {g['total']} | {rate} |\n"
    fp=Counter(); fn=Counter()
    for r in test:
        if int(r['true_label'])==int(r['pred_label']): continue
        if int(r['true_label'])==0:
            grade=''.join(c for c in r['原始评级'] if c.isalpha()).upper() or r['原始评级']
            fp[grade]+=1
        else: fn[r['异常代码'] or r['原始评级']]+=1
    text+='\n全部错误的主要分组（按数量，不代表组内错误率）：正常币误报 '+str(fp.most_common(5))+'；异常币漏检 '+str(fn.most_common(5))+'。\n'
    text+='\n分组小样本的错误率可能波动；这里仅作描述性分析。典型错误按错误置信度选取，包含编号、真实标签、整币与双面分数；未据测试结果调整模型或阈值。\n'
    text+=f'\n结果核验通过：融合、阈值、全部测试指标、最佳轮次、停止条件及固定划分哈希一致。Backbone 有 {changed}/{len(keys)} 个对比张量发生变化，确认实际进行了微调。详见结果/结果核验.json。\n'
    report=ROOT/'实验报告.md'; body=report_intro(config,result,history)
    report.write_text(body+text,encoding='utf-8')
    print(json.dumps({'checks':checks,'false_positive_groups':fp.most_common(),'false_negative_groups':fn.most_common()},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
