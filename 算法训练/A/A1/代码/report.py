import train as t
import fusion
import json,hashlib,csv
import numpy as np

def main():
    io=json.loads((t.OUT/'io_benchmark.json').read_text(encoding='utf-8'))
    f=json.loads((t.ROOT/'融合对照/results.json').read_text(encoding='utf-8'))
    a=json.loads((t.OUT/'test_metrics.json').read_text(encoding='utf-8'))
    h=json.loads((t.OUT/'history.json').read_text(encoding='utf-8'))
    c=json.loads((t.MODEL/'config.json').read_text(encoding='utf-8'))
    for k,v in t.FILES.items():assert hashlib.sha256((t.SPLIT/v).read_bytes()).hexdigest()==c['split_sha256'][k]
    rows=fusion.read(t.OUT/'test_predictions.csv'); y=np.array([int(r['true_label']) for r in rows]); p=np.array([float(r['coin_score']) for r in rows])
    for k,v in t.metrics(y,p,a['threshold']).items():assert v==a[k]
    assert len(rows)==755 and [r['钱币编号'] for r in rows]==[r['钱币编号'] for r in t.rows('test')]
    val=fusion.read(t.OUT/'val_predictions.csv'); vy=np.array([int(r['true_label']) for r in val]); vp=np.array([float(r['coin_score']) for r in val]); pr,re,ts=t.precision_recall_curve(vy,vp); fs=2*pr[:-1]*re[:-1]/(pr[:-1]+re[:-1]+1e-12)
    assert a['threshold']==float(ts[np.flatnonzero(fs==fs.max())[-1]])
    assert a['best_epoch']==max(h,key=lambda r:r['val_auroc'])['epoch']
    best=t.torch.load(t.MODEL/'best.pt',map_location='cpu',weights_only=False)['model']; initial=t.torch.load(t.ROOT.parent/'A0/模型/convnext_tiny-983f1562.pth',map_location='cpu',weights_only=True)
    changed=sum(not t.torch.equal(v,initial[k[4:]]) for k,v in best.items() if k.startswith('net.') and k[4:] in initial)
    assert changed>100
    t.save_json(t.OUT/'verification.json',dict(passed=True,split_hashes=c['split_sha256'],backbone_tensors_changed=changed,test_rows=len(rows),best_epoch=a['best_epoch'],threshold_source='validation only'))
    names={'max':('A0','Max Logit'),'front_back':('A0 后处理','正反面加权概率'),'max_mean':('A0 后处理','Max + Mean 加权')}
    table=[]
    for k,(name,method) in names.items():table.append(dict(experiment=name,model='ConvNeXt-Tiny',fusion=method,weight=f[k]['weight'],**{x:f[k]['test_metrics'][x] for x in ['accuracy','precision','recall','f1','auroc','ap']}))
    table.append(dict(experiment='A1',model='ConvNeXt-Tiny',fusion='双面特征拼接 + MLP',weight='',**{x:a[x] for x in ['accuracy','precision','recall','f1','auroc','ap']})); t.write_csv(t.OUT/'unified_comparison.csv',table)
    text=['# A1 双面特征融合实验报告','','## 方法与可比性','','A0 后处理仅使用已保存的正反面概率，没有训练或重新推理 A0。两种加权方法分别预先声明，在验证集搜索 0–1、步长 0.1，按 AUROC 选择，完全相同则按 AP、再按较小权重打破并列。权重选定后最大化验证 F1（并列选最高阈值）。原始 Max 保留 A0 阈值。测试集不参与选择。','','A1 从同一个 ImageNet 权重文件初始化，未加载 A0 best.pt。共享 ConvNeXt-Tiny，各面全局池化和原模型 LayerNorm 后得到 768 维向量，正面在前拼接为 1536 维，经 Linear(1536,256)、GELU、Linear(256,1) 输出整币 logit。全参数微调，BCEWithLogitsLoss。MLP 无额外 dropout 或注意力。','','输入仍为 384×384，独立 ±10° 双线性旋转、原有 ImageNet Normalize；缓存没有重采样、重新标准化或改写像素。AdamW、Batch 8、LR 2e-5/1e-4、WD .01、AMP、梯度裁剪 5、seed 42；15–50 轮、早停 10、验证 AUROC 选择。沿用 A0 的第 6 轮起 ReduceLROnPlateau 设置。持久 worker 使用共享 epoch 和样本编号确定增强随机种子，分布不变但随机轨迹与 A0 不完全相同，因此这是单种子对照，不能单独证明统计显著性。','','## 数据读取与真实测速','',f"缓存：`{io['cache']['manifest']}`；{io['cache']['files']} 个原始文件，{io['cache']['bytes']/2**30:.2f} GiB，复制前可用 {io['cache']['free_before']/2**30:.2f} GiB；全部 SHA256 一致。缓存耗时 {io['cache']['seconds']/60:.1f} 分钟。CSV 哈希、记录顺序与标签核验通过。",'',f"96 张样本串行阶段累计秒：原盘 {io['profile_original']}；SSD {io['profile_ssd']}。系统文件缓存未清空，因此不作为严格冷盘基准。",'','| 来源 | workers | pin | persistent | prefetch | 训练枚/秒 | 数据等待比例 | 启动及预热秒 |','|---|---:|---|---|---:|---:|---:|---:|']
    for r in io['benchmarks']:
        q=r['cfg'];text.append(f"| {r['source']} | {q['workers']} | {q['pin_memory']} | {q['persistent_workers']} | {q['prefetch_factor']} | {r['coins_per_second']:.2f} | {r['data_wait_fraction']:.1%} | {r['startup_warmup_seconds']:.1f} |")
    text+=['',f"所有吞吐测试使用 A1、固定样本和 4 步预热 + 32 步实际前反向/AdamW；选择配置 {io['selected']}。短测相对本次原盘基线加速 {io['speedup']:.2f} 倍。短测外推仅训练每轮 {io['estimated_train_epoch_seconds']:.1f} 秒；完整轮次另含验证与保存。",'',f"实际完成 {len(h)} 轮，最佳第 {a['best_epoch']} 轮；完整轮次平均 {a['training_seconds']/len(h):.1f} 秒，累计 {a['training_seconds']/3600:.2f} 小时，含最终评估墙钟 {a['wall_elapsed_seconds']/3600:.2f} 小时。实际训练吞吐 {6033*len(h)/sum(x['seconds'] for x in h):.2f} 枚/秒，数据等待占比 {sum(x['data_wait_seconds'] for x in h)/sum(x['seconds'] for x in h):.1%}。A0 历史平均 759.1 秒/轮，二者包含模型变化，不能把全部差异归因于缓存。",'','## 验证集融合选择','','| 方法 | 权重 | AUROC | AP | 阈值 | F1 |','|---|---:|---:|---:|---:|---:|']
    for k,r in f.items():text.append(f"| {k} | {r['weight']} | {r['val_auroc']:.6f} | {r['val_ap']:.6f} | {r['threshold']:.9f} | {r['val_metrics']['f1']:.6f} |")
    text+=['','全部 23 条验证记录见融合对照/validation_weight_search.csv。两种后处理方案的权重和阈值先固化再读取测试集；不能依据测试指标重新挑选方案。','', '## 固定测试集统一结果','','| 实验 | 模型 | 融合方式 | Accuracy | Precision | Recall | F1 | AUROC | AP |','|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in table:text.append('| '+ ' | '.join([r['experiment'],r['model'],r['fusion']]+[f"{r[k]:.6f}" for k in ['accuracy','precision','recall','f1','auroc','ap']])+' |')
    text+=['',f"A1 混淆矩阵（行真实、列预测）：{a['confusion_matrix']}；阈值 {a['threshold']:.9f}。",'','## 主要错误','','| 分组 | 总数 | A0 错误 | 正反面加权错误 | Max+Mean 错误 | A1 错误 |','|---|---:|---:|---:|---:|---:|']
    ag=fusion.groups(rows,p,a['threshold'])
    for i,g in enumerate(ag):text.append(f"| {g['group']} | {g['total']} | {f['max']['groups'][i]['errors']} | {f['front_back']['groups'][i]['errors']} | {f['max_mean']['groups'][i]['errors']} | {g['errors']} |")
    for k,r in f.items():
        if k=='max':continue
        base=f['max']['test_metrics'];m=r['test_metrics'];text+=['',f"{k} 相对原 Max：AUROC {m['auroc']-base['auroc']:+.6f}，AP {m['ap']-base['ap']:+.6f}，F1 {m['f1']-base['f1']:+.6f}；FP {base['confusion_matrix'][0][1]} → {m['confusion_matrix'][0][1]}，FN {base['confusion_matrix'][1][0]} → {m['confusion_matrix'][1][0]}。"]
    base=f['max']['test_metrics'];text+=['',f"A1 相对 A0：AUROC {a['auroc']-base['auroc']:+.6f}，AP {a['ap']-base['ap']:+.6f}，F1 {a['f1']-base['f1']:+.6f}；FP {base['confusion_matrix'][0][1]} → {a['confusion_matrix'][0][1]}，FN {base['confusion_matrix'][1][0]} → {a['confusion_matrix'][1][0]}。结合上表看正常 AU/XF/MS 误报与异常 92/97/98 漏检，不能仅凭 Accuracy 判断。",'','![训练曲线](结果/training_curves.png)','','![混淆矩阵](结果/confusion_matrix.png)','','完整预测、分组错误和典型错误见结果目录。恢复命令：使用模型/config.json 的 python 执行 `代码/train.py --mode train`；last.pt 保存完整轮次的模型、优化器、scaler 和 scheduler，未完成轮次重跑。训练完成后执行 `代码/report.py` 可重新整理报告，不再次推理测试集。']
    (t.ROOT/'实验报告.md').write_text('\n'.join(text),encoding='utf-8')
    import error_transitions
    error_transitions.main()
    from finalize_notes import append_notes
    append_notes()
    print('REPORT VERIFIED',flush=True)
if __name__=='__main__':main()
