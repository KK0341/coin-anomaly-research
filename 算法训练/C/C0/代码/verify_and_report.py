"""CPU-only audit of saved predictions and concise final reports; no model selection changes."""
import csv,json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
R=Path(__file__).resolve().parents[1]; EXP=R.parent
def read(p):return json.loads(p.read_text('utf8'))
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def near(a,b):assert np.isclose(a,b,rtol=1e-7,atol=1e-8),(a,b)
def pct(x):return f'{100*x:.2f}%'
def write(p,lines):p.write_text('\n'.join(lines)+'\n','utf8')
def main():
    labels=list(read(R/'label_mapping.json'));K=len(labels);counts=rows(R/'grade_counts.csv');configs={n:read(EXP/n/'config.json') for n in ['C0','C1']};summary={};audits={}
    for key in configs['C0']:
        if key not in ['experiment','initialization']:assert configs['C0'][key]==configs['C1'][key],key
    for name in configs:
        out=EXP/name;history=read(out/'history.json');complete=read(out/'COMPLETE.json');best=max(history,key=lambda h:h['val_macro_f1']);assert best['epoch']==complete['best_epoch']==read(out/'frozen_selection.json')['best_epoch'];assert history[-1]['stale']==10
        assert hashlib.sha256((Path(__file__).resolve().parent/'原始版本/train.py').read_bytes()).hexdigest()==configs[name]['code_sha256']
        init=read(out/'initialization.json');assert hashlib.sha256((Path(init['source']) if Path(init['source']).is_file() else R.parent.parent/'B/B0/数据缓存'/Path(init['source']).name).read_bytes()).hexdigest()==init['sha256']
        results={};flows={};checks={}
        for split in ['val','test']:
            rs=rows(out/f'{split}_all_pipeline_predictions.csv');metrics=read(out/f'{split}_metrics.json');flow=read(out/f'{split}_pipeline_metrics.json');normal=[r for r in rs if r['二分类标签']=='0'];ordinary=[r for r in rs if int(r['label'])>=0]
            assert len({r['证书编号'] for r in rs})==len(rs);assert len(normal)==metrics['normal_total']==556;assert len(ordinary)==metrics['n'];assert len(rows(out/f'{split}_predictions.csv'))==556
            cm=np.zeros((K,K),dtype=int);correct=0;direct=0;absolute=0;groupcm=np.zeros((4,4),dtype=int)
            for r in rs:
                probs=np.array([float(r['p_'+g]) for g in labels]);assert abs(probs.sum()-1)<1e-6;assert labels[probs.argmax()]==r['pred_grade'];assert (r['pipeline_pred']=='abnormal')==(r['A_pred_label']=='1')
                if int(r['label'])<0:continue
                y=int(r['label']);p=labels.index(r['pred_grade']);cm[y,p]+=1;direct+=int(y==p);correct+=int(r['pipeline_pred']==r['base_grade']);absolute+=abs(int(''.join(filter(str.isdigit,labels[y])))-int(''.join(filter(str.isdigit,labels[p]))))
                n=int(''.join(filter(str.isdigit,labels[y])));g=0 if n<=35 else 1 if n<=45 else 2 if n<=58 else 3;gp=np.array([float(r['p_group_'+x]) for x in ['VF及以下','XF','AU','MS']]);assert abs(gp.sum()-1)<1e-6;groupcm[g,gp.argmax()]+=1
            assert cm.tolist()==metrics['confusion_matrix'];assert groupcm.tolist()==metrics['four_group']['confusion_matrix'];near(np.trace(cm)/cm.sum(),metrics['accuracy']);near(absolute/cm.sum(),metrics['numeric_mae']);near(np.trace(groupcm)/cm.sum(),metrics['four_group']['accuracy'])
            fs=[]
            for i,g in enumerate(labels):
                tp=cm[i,i];pr=tp/cm[:,i].sum() if cm[:,i].sum() else 0;rec=tp/cm[i].sum() if cm[i].sum() else 0;f=2*pr*rec/(pr+rec) if pr+rec else 0;v=metrics['per_class'][g]
                near(pr,v['precision']);near(rec,v['recall']);near(f,v['f1']);assert int(cm[i].sum())==v['support'];fs.append(f)
            near(np.mean(fs),metrics['macro_f1']);near(correct/len(ordinary),flow['ordinary_end_to_end_accuracy']);near(correct/len(normal),flow['normal_all_denominator_correct_covered_rate']);assert sum(r['A_pred_label']=='1' for r in normal)==flow['normal_rejected_by_A'];assert sum(r['二分类标签']=='1' and r['A_pred_label']=='0' for r in rs)==flow['abnormal_entering_C']
            results[split]=metrics;flows[split]=flow;checks[split]=dict(coins=len(rs),normal=len(normal),ordinary=len(ordinary),direct_correct=direct,pipeline_correct=correct,metrics_recomputed=True,probabilities_checked=True)
        summary[name]=dict(best=best,history=history,results=results,flows=flows,checks=checks);audits[name]=checks
        v=results['val'];t=results['test'];f=flows['test'];gap=best['train_macro_f1']-best['val_macro_f1'];cm=np.array(v['confusion_matrix']);pairs=sorted([(int(cm[i,j]),labels[i],labels[j]) for i in range(K) for j in range(K) if i!=j and cm[i,j]],reverse=True)[:5]
        lines=[f'# {name} 实验报告','',f'目的：检验阶段一权重迁移对普通币基础数字等级分类的帮助。初始化：{init["transferred"]} 从 {Path(init["source"]).name} 迁移；{init["new"]} 重新初始化。全参数微调，训练中重新计算骨干特征。','',f'固定 Dataset V1 原始划分，普通监督币 train/val/test=4424/556/555，20类。真实正常币总数4444/556/556；SP/PR分别20/0/1，不参与普通等级监督；保留原始评级和plus，本轮不预测plus。','',f'最佳轮次 **{best["epoch"]}**，共训练 **{len(history)}** 轮，patience10正常早停。训练轮次耗时合计 {sum(h["seconds"] for h in history)/60:.1f} 分钟（不含初始化和最终评价）。','', '|集合|具体等级Accuracy|Macro-F1|数值MAE|四档Accuracy|四档Macro-F1|','|---|---:|---:|---:|---:|---:|']
        for s,m in results.items():lines.append(f'|{s}|{pct(m["accuracy"])}|{pct(m["macro_f1"])}|{m["numeric_mae"]:.3f}|{pct(m["four_group"]["accuracy"])}|{pct(m["four_group"]["macro_f1"])}|')
        lines+=['','Macro-F1固定按20类计算，零支持类F1记0；验证集G4/F12/F15/MS60无支持，测试集G4/F12/F15/VF20无支持。所有类别均有训练样本，没有重划分或删除稀有记录。四档通过细类概率求和后取最大值。','',f'验证主要混淆：'+ '；'.join(f'{a}→{b} {n}枚' for n,a,b in pairs)+'。',f'验证错误中相邻已观察等级占 {pct(v["adjacent_label_errors"]/v["errors"])}；最佳轮次训练Macro-F1={pct(best["train_macro_f1"])}，验证={pct(best["val_macro_f1"])}，差{100*gap:.2f}个百分点。训练指标来自增强数据和轮内模型，仅作为过拟合证据之一。','', '## 测试覆盖及A8完整流程','',f'- 独立C：普通数值等级正确 {checks["test"]["direct_correct"]}/555；全部556枚正常币均推理，含1枚SP/PR，无法评价其具体等级。以全部正常币作分母的已覆盖正确率为 {pct(t["all_normal_direct_correct_covered_rate"])}。',f'- A8冻结阈值 {f["threshold"]:.9f}；69/556正常币被送入异常分支（68枚普通币、1枚SP/PR），均不能成为完整流程的正确品相输出。',f'- A→C正确 {checks["test"]["pipeline_correct"]}/555普通币，准确率 {pct(f["ordinary_end_to_end_accuracy"])}；按全部正常币计 {checks["test"]["pipeline_correct"]}/556={pct(f["normal_all_denominator_correct_covered_rate"])}。',f'- 199枚异常币中71枚误入C；SP/PR 1枚，进入C 0枚；其他未覆盖正常类别0枚。',f'- 普通币流程Macro-F1（A拒绝计漏判）{pct(f["ordinary_macro_f1_with_A_rejections"])}；再将异常误入C计为各等级假阳性后，等级Macro-F1={pct(f["all_flow_grade_macro_f1"])}。','', '## 具体等级指标','', '|等级|训练支持|验证支持|验证P|验证R|验证F1|测试支持|测试P|测试R|测试F1|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for g,c in zip(labels,counts):
            a=v['per_class'][g];b=t['per_class'][g];lines.append(f'|{g}|{c["train"]}|{a["support"]}|{a["precision"]:.3f}|{a["recall"]:.3f}|{a["f1"]:.3f}|{b["support"]}|{b["precision"]:.3f}|{b["recall"]:.3f}|{b["f1"]:.3f}|')
        lines+=['','配置：seed42，512输入，batch4×累积2，BF16，±10°旋转；AdamW主干1e-5/头1e-4，weight decay0.01；C训练集逆平方根类别权重；验证Macro-F1平台降学习率。验证选模后冻结，测试未用于选择配置。','', '文件：best.pt、config.json、label_mapping.json、history.csv、training_curves.png；val/test_predictions.csv含所有真实正常币全类概率，all_pipeline_predictions含全测试集；混淆矩阵按基础数值排序。','', 'A3四档参考：验证77.34%、测试78.38%，按异常AUROC选模，与本轮具体等级任务不同。']
        write(out/'实验报告.md',lines)
    a=summary['C0'];b=summary['C1'];delta=a['results']['val']['macro_f1']-b['results']['val']['macro_f1'];lines=['# C0/C1 对比报告','', '两组训练、正常早停及最终评价均完成。已从逐币概率重新核验Accuracy、Macro-F1、各类P/R/F1、数值MAE、四档混淆矩阵及流程准确率；配置除初始化外一致。','', '|指标|C0：A8迁移|C1：ImageNet|','|---|---:|---:|',f'|最佳轮次 / 实际轮次|{a["best"]["epoch"]} / {len(a["history"])}|{b["best"]["epoch"]} / {len(b["history"])}|']
    for split,title in [('val','验证'),('test','测试')]:
        for key,label in [('accuracy','具体等级Accuracy'),('macro_f1','具体等级Macro-F1')]:lines.append(f'|{title}{label}|{pct(a["results"][split][key])}|{pct(b["results"][split][key])}|')
        lines.append(f'|{title}四档Accuracy|{pct(a["results"][split]["four_group"]["accuracy"])}|{pct(b["results"][split]["four_group"]["accuracy"])}|')
    lines += [f'|测试数值MAE|{a["results"]["test"]["numeric_mae"]:.3f}|{b["results"]["test"]["numeric_mae"]:.3f}|',f'|A→C测试正确 / 全部正常币|{a["checks"]["test"]["pipeline_correct"]}/556（{pct(a["flows"]["test"]["normal_all_denominator_correct_covered_rate"])}）|{b["checks"]["test"]["pipeline_correct"]}/556（{pct(b["flows"]["test"]["normal_all_denominator_correct_covered_rate"])}）|','',f'**阶段一初始化有小幅帮助，证据有限。** 按事先固定的验证Macro-F1，C0较C1高 {delta*100:.2f} 个百分点。测试Accuracy高 {(a["results"]["test"]["accuracy"]-b["results"]["test"]["accuracy"])*100:.2f} 个百分点、Macro-F1高 {(a["results"]["test"]["macro_f1"]-b["results"]["test"]["macro_f1"])*100:.2f} 个百分点。测试仅作冻结后的评价；单一种子不能证明稳定显著优势。','', '**错误主要集中在相邻等级。** C0验证错误中67.8%为相邻已观察等级，主要是MS64→MS63、MS62↔MS63、XF45→XF40及AU50→AU53。“相邻”按实际类别数值排序；数值差≤3的错误占61.4%。','',f'**过拟合是首要问题，少数类也限制评价。** 最佳轮次训练/验证Macro-F1：C0 {pct(a["best"]["train_macro_f1"])}/{pct(a["best"]["val_macro_f1"])}，C1 {pct(b["best"]["train_macro_f1"])}/{pct(b["best"]["val_macro_f1"])}。两组最后训练Macro-F1均超过98%，验证仍约24%。训练<20的六类只有3枚验证样本，C0这些类验证F1均为0；这不能解释为每个稀有类均已被充分评价。','', '**下一轮优先减少可训练参数，其次尝试有序分类。** 先对C0初始化做部分冻结对照，针对明显过拟合；再验证有序损失能否改善相邻混淆。局部细节应在误判图像分析后决定，当前结果不能证明分辨率或局部纹理不足是主要瓶颈。本轮不启动新实验。','', '覆盖与分母：普通监督币4424/556/555；真实正常币4444/556/556，另21枚SP/PR（20/0/1）。20类都有训练支持；验证和测试各有4类零支持，固定20类Macro-F1包含这些零值，不更改选模口径。plus合并到基础等级但保留标记，不预测plus。','', 'A8流程：冻结阈值0.306527048，测试69/556真实正常币误送异常分支，其中68枚普通币、1枚SP/PR；199枚异常币中71枚误入C。完整流程中的正常币失败均保留；异常误入C另计并纳入全流程等级precision。详见各组报告。','', 'A3四档参考：验证77.34%、测试78.38%；C0为81.12%/82.34%，C1为76.62%/78.56%。A3按异常AUROC选模，训练目标不同，仅作粗品相参考。','', '文件：C0/实验报告.md、C1/实验报告.md、verification.json。最佳权重、配置、曲线、全概率逐币预测及混淆矩阵均在各自目录。']
    write(R/'C0_C1对比报告.md',lines)
    report=dict(status='passed',config_equal_except_initialization=True,code_and_initialization_hashes_verified=True,best_epoch_matches_validation_history=True,normal_early_stopping=True,experiments=audits)
    (R/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),'utf8');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
