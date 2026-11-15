from pathlib import Path
import json,hashlib,re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,precision_recall_curve
O=Path(__file__).resolve().parents[1]; R=O.parent
INPUTS={}
def read(p):
 INPUTS[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest(); return pd.read_csv(p,dtype=str).fillna('')
def js(p):
 INPUTS[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest(); return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 d=read(Path('E:/异常检测/数据划分/验证集.csv')); y=d['二分类标签'].astype(int).to_numpy(); ids=d['钱币编号']; assert len(d)==755 and sum(y)==199 and ids.is_unique
 def aligned(path,col='coin_score'):
  f=read(path); assert f['钱币编号'].is_unique and set(f['钱币编号'])==set(ids); f=f.set_index('钱币编号').loc[ids]
  for k in ['正面路径','背面路径','二分类标签']:
   if k in f: assert f[k].tolist()==d[k].tolist()
  if 'true_label' in f: assert np.array_equal(f.true_label.astype(int),y)
  s=f[col].astype(float).to_numpy(); assert np.isfinite(s).all(); return s
 scores={'A1-384':aligned(R/'A1/结果/val_predictions.csv'),'A8-512':aligned(R/'A8_高分辨率实验/结果/val_predictions.csv'),'A10b-DINOv2-504':aligned(O/'结果/val_predictions.csv')}
 def threshold(s,point):
  if point=='recall154': return float(np.sort(s[y==1])[::-1][153])
  p,r,t=precision_recall_curve(y,s); f=2*p[:-1]*r[:-1]/(p[:-1]+r[:-1]+1e-12); return float(t[np.flatnonzero(f==f.max())[-1]])
 def measure(s,point):
  t=threshold(s,point); pr=s>=t; tp=int(sum(pr&(y==1))); fp=int(sum(pr&(y==0))); fn=199-tp; tn=556-fp
  return dict(AUROC=roc_auc_score(y,s),AP=average_precision_score(y,s),threshold=t,TP=tp,FP=fp,FN=fn,TN=tn,Accuracy=(tp+tn)/755,Precision=tp/(tp+fp),Recall=tp/199,F1=2*tp/(2*tp+fp+fn),exact_TP154=tp==154),pr
 a8,_=measure(scores['A8-512'],'recall154'); assert a8['FP']==104 and abs(a8['AUROC']-.865185)<.0000005 and abs(a8['AP']-.721563)<.0000005
 a10,pr10=measure(scores['A10b-DINOv2-504'],'recall154'); pr8=measure(scores['A8-512'],'recall154')[1]
 masks={g:(y==1)&d['异常代码'].str.contains(r'(?<!\d)'+g+r'(?!\d)',regex=True).to_numpy() for g in ['92','97','98']}; masks['其他异常']=(y==1)&~(masks['92']|masks['97']|masks['98'])
 for g in ['AU','XF']: masks['正常'+g]=(y==0)&d['原始评级'].str.startswith(g).to_numpy()
 gate=a10['AUROC']-a8['AUROC']>=.005 or (a10['TP']==154 and (a8['FP']-a10['FP']>=5 or sum(~pr8&masks['92'])-sum(~pr10&masks['92'])>=3))
 if gate:
  cfg=js(R/'A5/frozen_validation_config.json'); ec=js(R/'A1/融合对照/A0_A1集成/frozen_ensemble_config.json'); alpha=ec['alpha']; w=cfg['fusion']['A0+A1+A5b']['weight']; assert alpha==.2 and w==.1
  a0=aligned(R/'A0/结果/val_predictions.csv'); a5=aligned(R/'A5/结果/local_scores.csv','A5b')
  def cdf(s,k):
   k=np.sort(k); return (np.searchsorted(k,s,'left')+np.searchsorted(k,s,'right'))/(2*len(k))
  new=alpha*a0+(1-alpha)*scores['A10b-DINOv2-504']; scores['A0+A10b+A5b']=(1-w)*cdf(new,cfg['cdf_knots']['A0+A1'])+w*cdf(a5,cfg['cdf_knots']['A5b'])
  scores['A0+A8+A5b']=aligned(R/'A8_高分辨率实验/结果/aligned_val_predictions.csv','A0+A8+A5b')
 save(O/'结果/fusion_decision.json',dict(gate_passed=bool(gate),protocol=js(O/'protocol.json'),no_weight_search=True))
 rows=[]; groups=[]; transitions=[]; errors=[]; pred=d[['钱币编号','证书编号','二分类标签','异常代码','原始评级','正面路径','背面路径']].copy()
 for name,s in scores.items():
  pred[name+'_score']=s
  for point in ['recall154','maxF1']:
   metric,pr=measure(s,point); rows.append(dict(model=name,point=point,**metric)); pred[name+'_'+point]=pr.astype(int)
   for group,mask in masks.items(): groups.append(dict(model=name,point=point,group=group,n=int(sum(mask)),errors=int(sum(mask&(pr!=y))),error_rate=float(sum(mask&(pr!=y))/sum(mask))))
   for i in np.flatnonzero(pr!=y): errors.append(dict(model=name,point=point,coin_id=ids.iloc[i],error='FN' if y[i] else 'FP',code=d['异常代码'].iloc[i],grade=d['原始评级'].iloc[i],score=s[i]))
   refs=['A1-384','A8-512'] if name=='A10b-DINOv2-504' else ['A0+A8+A5b'] if name=='A0+A10b+A5b' else []
   for ref in refs:
    old=measure(scores[ref],point)[1]
    for i in np.flatnonzero(pr!=old): transitions.append(dict(model=name,reference=ref,point=point,coin_id=ids.iloc[i],change=('rescued_FN' if pr[i] else 'new_FN') if y[i] else ('new_FP' if pr[i] else 'removed_FP'),code=d['异常代码'].iloc[i],grade=d['原始评级'].iloc[i],old_score=scores[ref][i],new_score=s[i]))
 for name,data in [('metrics_comparison.csv',rows),('group_errors.csv',groups),('error_transitions.csv',transitions),('error_samples.csv',errors),('aligned_val_predictions.csv',pred)]: pd.DataFrame(data).to_csv(O/'结果'/name,index=False,encoding='utf-8-sig')
 history=js(O/'结果/history.json'); config=js(O/'模型/config.json'); best=js(O/'模型/threshold.json'); smoke=js(O/'结果/smoke.json'); source=js(O/'来源/provenance.json'); loading=js(O/'结果/weight_loading.json')
 assert best['best_epoch']==max(history,key=lambda h:h['val_auroc'])['epoch']; assert abs(best['val_metrics']['auroc']-a10['AUROC'])<1e-12
 timings=[]
 for name,path in [('A1-384',R/'A1/结果'),('A8-512',R/'A8_高分辨率实验/结果'),('A10b-DINOv2-504',O/'结果')]:
  h=js(path/'history.json'); b=max(h,key=lambda r:r['val_auroc']); timings.append(dict(model=name,epochs=len(h),best_epoch=b['epoch'],training_minutes=sum(r['epoch_total_seconds'] for r in h)/60,peak_allocated_MiB=max(r['allocated_MiB'] for r in h)))
 pd.DataFrame(timings).to_csv(O/'结果/training_cost.csv',index=False,encoding='utf-8-sig')
 def table(frame,cols):
  out=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
  for _,r in pd.DataFrame(frame).iterrows(): out.append('| '+' | '.join(f'{r[c]:.6f}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in cols)+' |')
  return '\n'.join(out)
 gdf=pd.DataFrame(groups); mdf=pd.DataFrame(rows); delta=a10['AUROC']-a8['AUROC']; dfp=a10['FP']-a8['FP']; dap=a10['AP']-a8['AP']
 verdict='在AUROC、AP及主工作点误报上均超过A8' if delta>0 and dap>0 and dfp<0 and a10['TP']==154 else '未同时改善A8的AUROC、AP和主工作点误报，应逐项权衡'
 lines=['# A10：预训练视觉骨干对照实验','',f'完成一个新骨干的全参数微调：A10b DINOv2 ViT-S/14，无Registers。{verdict}。', '',
 '## 来源及控制','',
 'DINOv3未成功使用：本机默认Torch/Hugging Face无权重缓存，原环境未安装timm；官方Meta渠道要求申请授权，官方Hugging Face模型文件实际返回HTTP401 restricted access。没有反复重试或使用非官方镜像，自动切换官方公开DINOv2。原始访问检查见来源/access_checks.json。',
 '[DINOv3官方项目](https://github.com/facebookresearch/dinov3)；[DINOv2官方项目](https://github.com/facebookresearch/dinov2)。',
 f"官方代码commit：`{source['commit']}`；[权重来源]({source['weight_url']})；SHA256：`{source['sha256']}`。权重strict=True加载，{loading['keys']}个键全部匹配，加载后逐张量相等；骨干参数{loading['backbone_parameters']:,}，总参数{loading['total_parameters']:,}，不是随机初始化或部分加载。",
 '相对A8：ConvNeXt-Tiny替换为ViT-S/14，单面全局特征为官方forward输出的normalized CLS 384维（原768），共享骨干处理正背面，拼接768维后进入Linear(768,256)→GELU→Linear(256,1)。输入504×504可被patch14整除（36×36 patch），与原512相差8像素。这是架构、预训练来源、特征和小幅分辨率变化的联合对照，不能归因为纯预训练收益。',
 '保留：原训练/验证划分，seed42，逐样本旋转±10°和ImageNet归一化，整币BCE，全参数AdamW（骨干2e-5、头1e-4、weight_decay0.01），梯度裁剪5，FP16，batch8，min15/max50轮，早停patience10，验证AUROC选模及原ReduceLROnPlateau策略。验证最大F1并列取最高阈值。',
 '资源：仅使用动态选择的一张空闲GPU；训练loader使用2 workers、验证0 workers以避免既有Windows内存问题（A8原4 workers）。官方实现使用PyTorch SDPA，无xFormers；未改变官方网络代码。预检先micro2×累积4，再验证micro8×累积1可行，正式batch8，无梯度检查点。所有骨干参数requires_grad=True；未屏蔽图像时mask_token不参与计算，没有人为冻结骨干。早期patch、首块、末块和最终norm均有非零有效梯度并实测更新，预检模型随后丢弃，正式重新seed及加载。',
 f"正式最佳轮次{best['best_epoch']}，完成{len(history)}轮。训练时长和显存见下表，预检峰值{smoke['allocated_MiB']:.1f} MiB。",'',table(timings,['model','epochs','best_epoch','training_minutes','peak_allocated_MiB']), '',
 '## 同755枚验证集结果','',f'正常556、异常199。相对A8：AUROC差{delta:+.6f}，AP差{dap:+.6f}，固定召回FP差{dfp:+d}。',
 'recall154取异常第154高分作为阈值，同分整体处理，exact_TP154列验证是否真正同召回；若为False，不以该点宣称误报改善。maxF1为各自验证最优点，不能替代同召回比较。','',table(rows,['model','point','AUROC','AP','Accuracy','Precision','Recall','F1','TP','FP','FN','exact_TP154']), '',
 '## 重点分组','', '异常组错误为FN，正常AU/XF为FP。','',table(groups,['model','point','group','n','errors','error_rate']), '',
 '## 实例变化','']
 tr=pd.DataFrame(transitions)
 for ref in ['A1-384','A8-512']:
  sub=tr[(tr.model=='A10b-DINOv2-504')&(tr.reference==ref)&(tr.point=='recall154')] if len(tr) else tr
  counts=sub.change.value_counts().to_dict() if len(sub) else {}
  lines.append(f'相对{ref}主工作点：{counts}。编号、异常代码、评级及前后分数已保存error_transitions.csv；所有错误另存error_samples.csv。')
 lines+=['','## 固定融合与研究判断','',f'训练前冻结门槛：AUROC至少提高0.005，或同召回FP减少至少5，或92漏检减少至少3。本次门槛通过={gate}。']
 if gate: lines+=['仅做一次规则复用：0.2A0+0.8A10b，再通过A5原有A0+A1 CDF节点映射，与原A5b CDF以0.9/0.1融合。保留旧节点，无新校准和权重搜索。跨骨干概率尺度未独立校准，该固定映射的效果也需独立确认。']
 else: lines+=['门槛未通过，不追加融合。']
 for g in masks:
  old=int(sum((pr8!=y)&masks[g])); new=int(sum((pr10!=y)&masks[g])); lines.append(f'{g}：{old}→{new}枚错误，'+('改善' if new<old else '退步' if new>old else '持平')+'。')
 lines += ['','组合参考A0+A8+A5b AUROC/AP=0.877386/0.738487、FP88；A9固定融合A4a=0.883288/0.752145、FP85，仅作为组合参考，不与本轮单模型进行架构优劣归因。',
 '局部正常参照比较尚未实验：全局整币分类收益不能证明patch级距离更适合钱币局部缺陷。'+('若主线收益可复现，可保留为后续局部特征小规模对照候选。' if delta>0 and dfp<=0 else '目前不应优先扩大到局部参照实验，应先解决整币主工作点问题。'),
 '保留本次最佳模型及全部预测。'+('有理由开展严格配对种子复现及新独立数据确认，重点审查92与正常AU/XF之间的取舍。' if delta>0 or dfp<0 else '当前没有充分理由投入更多大型训练，优先保留基线并检查本轮错误模式。'),
 '本轮仅一个新骨干、一个种子，验证集参与选模且已被多次探索；不声称独立测试结论或统计显著提升。未读取测试集、未重训A1/A8、未增加原因头/专家模块、未清理原缓存。',
 '', '代码/launch.py可在无空闲GPU时后台排队；代码/train.py支持last.pt断点继续；训练结束自动运行evaluate.py完成验证对照与报告。来源、配置、SHA256、smoke检查和逐币预测均保存在本目录。']
 (O/'实验报告.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 for p,h in js(O/'来源/protected_input_hashes.json').items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 for p,h in INPUTS.items(): assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 save(O/'结果/evaluation_input_hashes.json',INPUTS); save(O/'结果/COMPLETE.json',dict(status='complete',variant='A10b',a8=a8,a10=a10,fusion_gate=bool(gate),n=755,all_input_hashes_unchanged=True))
 print('A10_COMPLETE '+json.dumps(a10),flush=True)
if __name__=='__main__': main()
