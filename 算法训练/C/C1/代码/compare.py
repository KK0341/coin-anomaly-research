import json,csv
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parents[1]; EXP=R.parent
def read(p):return json.loads(p.read_text('utf8'))
def main():
    results={n:read(EXP/n/'COMPLETE.json') for n in ['C0','C1']};configs={n:read(EXP/n/'config.json') for n in results}
    for k in configs['C0']:
        if k not in ['experiment','initialization']:assert configs['C0'][k]==configs['C1'][k],k
    delta=results['C0']['metrics']['val']['macro_f1']-results['C1']['metrics']['val']['macro_f1'];winner='C0' if delta>0 else 'C1' if delta<0 else 'tie'
    lines=['# C0 / C1 对比报告','固定 Dataset V1、seed42、共享 ConvNeXt-Tiny 双面结构；仅初始化来源不同。两组配置一致性已核验。验证 Macro-F1 选模并比较方案，测试用于冻结后的最终评价。','|实验|最佳轮次|验证Accuracy|验证Macro-F1|测试Accuracy|测试Macro-F1|测试MAE|测试四档Accuracy|','|---|---:|---:|---:|---:|---:|---:|---:|']
    rows=[]
    for n,r in results.items():
        v=r['metrics']['val'];t=r['metrics']['test'];lines.append(f'|{n}|{r["best_epoch"]}|{v["accuracy"]:.4f}|{v["macro_f1"]:.4f}|{t["accuracy"]:.4f}|{t["macro_f1"]:.4f}|{t["numeric_mae"]:.3f}|{t["four_group"]["accuracy"]:.4f}|');rows.append(dict(experiment=n,best_epoch=r['best_epoch'],val_macro_f1=v['macro_f1'],test_macro_f1=t['macro_f1'],test_accuracy=t['accuracy']))
    lines += [f'阶段一初始化：C0−C1 验证 Macro-F1 = {delta:+.4f}；本次固定种子下，'+('有帮助。' if delta>0 else '未显示帮助。')+'单次实验不推断统计显著性。']
    selected='C0' if winner=='tie' else winner;v=results[selected]['metrics']['val'];hist=read(R/selected/'history.json');best=next(h for h in hist if h['epoch']==results[selected]['best_epoch']);gap=best['train_macro_f1']-best['val_macro_f1'];adj=v['adjacent_label_errors']/max(v['errors'],1)
    rare=[g for g,n in zip(configs[selected]['classes'],configs[selected]['class_counts']) if n<20];rare_support=sum(v['per_class'][g]['support'] for g in rare);rare_f1=float(np.mean([v['per_class'][g]['f1'] for g in rare])) if rare else 0
    lines += [f'验证较优方案 {selected}：错误中相邻已观察等级占 {adj:.1%}，数值差≤3占 {v["within_3_numeric_errors"]/max(v["errors"],1):.1%}。相邻是按固定数值顺序的类别索引定义。',f'少数类（训练<20）：{rare}；验证支持合计 {rare_support}，平均F1 {rare_f1:.4f}。缺失训练支持类别：{configs[selected]["unsupported_train"]}，未删除、未重划分。',f'最佳轮次训练−验证 Macro-F1 差 {gap:.4f}。训练指标来自增强图及训练中模型，差值仅为过拟合线索；同时查看完整训练曲线。']
    recommendation='减少可训练参数，控制过拟合' if gap>.2 else '有序分类，利用相邻品相关系' if adj>.5 else '局部细节建模，先检查币面误判样本'
    lines += [f'下一轮优先建议：{recommendation}。基于验证结果提出，不自动启动新实验；少数类支持不足会限制结论。','两组每类P/R/F1、支持量、顺序混淆、全类别概率及A8流程统计见各自文件夹。所有真实正常币直接进入C；普通数值等级指标排除SP/PR并报告覆盖；全正常流程分母包含SP/PR，A误送异常分支计失败，异常误入C单列且计入全流程等级precision。','A3 四档参考：验证77.34%、测试78.38%；A3按异常AUROC选模，与本轮具体等级任务不同，不能作为严格公平对照。','本轮C0/C1已完成，后续方案依据本报告决定。']
    (R/'C0_C1对比报告.md').write_text('\n\n'.join(lines),'utf8')
    with (R/'C0_C1对比.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (R/'COMPLETE.json').write_text(json.dumps(dict(validation_winner=winner,delta_val_macro_f1=delta,recommendation=recommendation),ensure_ascii=False,indent=2),'utf8')
if __name__=='__main__':main()
