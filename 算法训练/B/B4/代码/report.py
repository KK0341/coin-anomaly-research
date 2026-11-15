import json,csv
from pathlib import Path
import numpy as np
import b0_reference as b
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text('utf8'))
def main():
    complete=read(R/'COMPLETE.json');best=complete['best_epoch'];history=read(R/'history.json');feature=read(R/'feature_analysis.json')
    experiments=['B0','B2','B3','B4'];metrics={e:{s:read(R.parent/e/f'{s}_metrics.json') for s in ['val','test']} for e in experiments}
    table=[];perclass=[];conf=[]
    for e in experiments:
        h=read(R.parent/e/'history.json');table.append(dict(experiment=e,val_macro_f1=metrics[e]['val']['macro_f1'],test_macro_f1=metrics[e]['test']['macro_f1'],epochs=len(h),best_epoch=max(h,key=lambda x:x['val_macro_f1'])['epoch'],minutes=sum(x['seconds'] for x in h)/60))
        for s in ['val','test']:
            for c in b.CODES:perclass.append(dict(experiment=e,split=s,code=c,**metrics[e][s]['per_class'][c]))
            for a,c in [('98','92'),('92','98'),('95','92'),('97','94')]:conf.append(dict(experiment=e,split=s,true=a,pred=c,count=metrics[e][s]['confusion_matrix'][b.CODES.index(a)][b.CODES.index(c)]))
    b.csvwrite(R/'B0_B2_B3_B4对比.csv',table);b.csvwrite(R/'六分类指标对比.csv',perclass);b.csvwrite(R/'主要混淆对比.csv',conf)
    frows=[]
    for e in ['B0','B4']:
        for s in ['train','val']:
            for c,v in feature[e][s].items():frows.append(dict(experiment=e,split=s,code=c,**v))
    b.csvwrite(R/'特征分布统计.csv',frows)
    ch=[json.loads(x) for x in (R/'center_history.jsonl').read_text('utf8').splitlines()];centerrows=[]
    for h in ch:
        for i,c in enumerate(b.CODES):centerrows.append(dict(epoch=h['epoch'],code=c,**{k:h[k][i] for k in ['class_center_loss','summed_gradient_norm','parameter_update_norm','center_norms','distance_from_initial']}))
    b.csvwrite(R/'类别中心训练记录.csv',centerrows)
    lines=['# B4 实验报告','',f'最佳第{best}轮，共{len(history)}轮；仅本机RTX 5060。验证Macro-F1={metrics["B4"]["val"]["macro_f1"]*100:.2f}%，相对B0变化{(metrics["B4"]["val"]["macro_f1"]-metrics["B0"]["val"]["macro_f1"])*100:+.2f}个百分点。','',
    '唯一改动：在B0双面512 ConvNeXt-Tiny融合后的256维GELU输出上增加六个可学习类别中心。特征和中心L2归一化，平方欧氏距离按样本平均，L=原逆平方根加权CE+0.1×中心损失。中心不参与分类前向，不添加类别间排斥项。','',
    '从A8原始权重重新初始化骨干与隐藏层；六分类输出与B0同seed42初始化，未加载B0/B3已训练参数。中心独立生成器seed42004初始化为单位随机向量，不改变网络随机序列。AdamW骨干1e-5、头1e-4、weight decay .01；中心独立组lr1e-4、weight decay0（归一化方向约束无需尺度衰减），同平台调度降至最低1e-6。中心计算为float32，网络BF16；batch4×累积2。CE按累积组类别权重之和归一化，中心损失按累积组样本数归一化，末组同样处理。','',
    '固定1583/198/198枚、六类92/97/98/95/94/91、原双面配对和±10°旋转。4个数据worker；全骨干可训练；最大60轮，patience10，严格验证Macro-F1提高才保存，平局保留早轮。梯度裁剪5。新增1536个训练参数；inference.pt移除中心，部署网络与B0一致。','',
    '|实验|验证Macro-F1|测试Macro-F1|最佳轮|训练分钟|','|---|---:|---:|---:|---:|']
    for r in table:lines.append(f'|{r["experiment"]}|{r["val_macro_f1"]*100:.2f}%|{r["test_macro_f1"]*100:.2f}%|{r["best_epoch"]}|{r["minutes"]:.1f}|')
    for s in ['val','test']:
        lines+=['',f'## {s} 六类指标（%）','','|类别|B0 P/R/F1|B4 P/R/F1|F1变化百分点|','|---|---:|---:|---:|']
        for c in b.CODES:
            a=metrics['B0'][s]['per_class'][c];v=metrics['B4'][s]['per_class'][c]
            lines.append(f'|{c} {v["name"]}|'+ ' / '.join(f'{a[k]*100:.2f}' for k in ['precision','recall','f1'])+'|'+ ' / '.join(f'{v[k]*100:.2f}' for k in ['precision','recall','f1'])+f'|{(v["f1"]-a["f1"])*100:+.2f}|')
        lines+=['','|混淆|B0|B4|','|---|---:|---:|']
        for a,c in [('98','92'),('92','98'),('95','92'),('97','94')]:lines.append(f'|{a}→{c}|'+ '|'.join(str(metrics[e][s]['confusion_matrix'][b.CODES.index(a)][b.CODES.index(c)]) for e in ['B0','B4'])+'|')
    train=read(R/'train_metrics.json');bh=next(h for h in history if h['epoch']==best);last=history[-1]
    lines+=['','## 拟合与中心更新','',f'最佳权重无增强训练Macro-F1 {train["macro_f1"]*100:.2f}%，验证{metrics["B4"]["val"]["macro_f1"]*100:.2f}%；最后轮在线训练F1 {last["train_macro_f1"]*100:.2f}%，验证{last["val_macro_f1"]*100:.2f}%。最佳轮验证CE {bh["val_loss"]:.4f}，最后轮{last["val_loss"]:.4f}。在线训练含增强及轮内参数变化，不能直接等同最佳权重无增强推理。','',f'中心损失：首轮{history[0]["train_center_loss"]:.4f}→最佳轮{bh["train_center_loss"]:.4f}→末轮{last["train_center_loss"]:.4f}。各类别每轮梯度累计范数、更新幅度、中心范数与方向变化见类别中心训练记录.csv；完整中心余弦矩阵见center_history.jsonl。正式峰值已分配显存{max(h["peak_memory_MiB"] for h in history):.1f}MiB。','',
    '## 特征分布（不等同分类提升）','','先对隐藏特征L2归一化；在各自模型的训练集计算每类经验均值方向，再计算训练/验证样本到对应训练均值方向的平方距离。余弦margin为真实类相似度减去最大其他类相似度；负值表示更接近另一类。两个模型均使用最佳权重无增强特征。','','|类|B0验证类内距离|B4验证类内距离|B0验证余弦margin|B4验证余弦margin|','|---|---:|---:|---:|---:|']
    for c in b.CODES:
        a=feature['B0']['val'][c];v=feature['B4']['val'][c];lines.append(f'|{c}|{a["mean_squared_distance_to_train_centroid"]:.4f}|{v["mean_squared_distance_to_train_centroid"]:.4f}|{a["mean_cosine_margin"]:.4f}|{v["mean_cosine_margin"]:.4f}|')
    lines+=['','feature_PCA.png分别在每个模型自己的训练特征上拟合PCA后投影训练/验证，坐标系不同，不能直接比较位置或二维聚集来声称性能提高。center_cosine.png区分经验均值与可学习中心。中心约束只拉近真实类，没有直接推开异类，因此类内距离减小仍可能伴随类别混杂。','',
    '## 结果性质','','方法、0.1权重和选模规则在训练前固定，最佳权重固定后测试仅推理一次并缓存；未据测试结果调参。历史测试集多次使用，仅作描述性对比。验证参与选模，单seed不支持统计显著性结论。没有启动B5或搜索中心损失权重。B0/B1/B2/B3已有文件未修改；B根部由C2运行器持有的历史锁在释放后归档到B4/运行记录。']
    (R/'实验报告.md').write_text('\n'.join(lines),'utf8')
    print(json.dumps(table,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
