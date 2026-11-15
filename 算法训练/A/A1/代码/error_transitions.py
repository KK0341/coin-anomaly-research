import train as t
import fusion,json
import numpy as np

def main():
    base=fusion.read(t.ROOT.parent/'A0/结果/test_predictions.csv'); byid={r['钱币编号']:r for r in base}
    variants={k:(fusion.read(t.ROOT/f'融合对照/{k}_test_predictions.csv'),'fusion_pred','fusion_score') for k in ['front_back','max_mean']}
    if (t.OUT/'test_predictions.csv').exists():variants['A1']=(fusion.read(t.OUT/'test_predictions.csv'),'pred_label','coin_score')
    summary=[]; transitions=[]
    for name,(rows,predkey,scorekey) in variants.items():
        for label in [0,1]:
            n={'method':name,'true_label':label,'corrected':0,'new_errors':0,'persistent_errors':0,'both_correct':0}
            for r in rows:
                if int(r['true_label'])!=label:continue
                b=byid[r['钱币编号']]; old=int(b['pred_label'])!=label; new=int(r[predkey])!=label
                kind='persistent_errors' if old and new else 'corrected' if old else 'new_errors' if new else 'both_correct'
                n[kind]+=1
                if old or new:transitions.append({'method':name,'coin_id':r['钱币编号'],'true_label':label,'grade':r['原始评级'],'abnormal_codes':r['异常代码'],'transition':kind,'A0_score':b['coin_score'],'new_score':r[scorekey]})
            summary.append(n)
    t.write_csv(t.OUT/'error_transitions_summary.csv',summary);t.write_csv(t.OUT/'error_transitions.csv',transitions)
    report=t.ROOT/'实验报告.md'
    if report.exists():
        text=report.read_text(encoding='utf-8').split('\n## 错误迁移')[0]
        text+='\n## 错误迁移\n\n| 方法 | 真实标签 | 修正 A0 错误 | 新增错误 | 共同错误 |\n|---|---:|---:|---:|---:|\n'
        for r in summary:text+=f"| {r['method']} | {r['true_label']} | {r['corrected']} | {r['new_errors']} | {r['persistent_errors']} |\n"
        text+='\n标签 0 为正常币误报，标签 1 为异常币漏检。逐币对照见结果/error_transitions.csv；本分析仅解释固定测试结果，不用于重新选权重或阈值。\n'
        report.write_text(text,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
