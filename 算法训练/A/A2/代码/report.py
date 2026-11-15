"""Report only: reads fixed predictions; never fits or re-evaluates models."""
import train as t
import json,csv,hashlib,time
import numpy as np
R=t.ROOT;O=t.OUT

def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))
def calc(rs):
 y=np.array([int(r['true_label']) for r in rs]);p=np.array([float(r['coin_score']) for r in rs]);pred=np.array([int(r['pred_label']) for r in rs])
 if not len(y):return {'n':0}
 return dict(n=len(y),normal=int(sum(y==0)),abnormal=int(sum(y==1)),accuracy=float(np.mean(y==pred)),precision=t.precision_score(y,pred,zero_division=0),recall=t.recall_score(y,pred,zero_division=0),f1=t.f1_score(y,pred,zero_division=0),auroc=t.roc_auc_score(y,p) if len(set(y))==2 else None,ap=t.average_precision_score(y,p) if sum(y)>0 else None,fp=int(sum((y==0)&(pred==1))),fn=int(sum((y==1)&(pred==0))))
def table(rs,keys):
 def fmt(x):return f'{x:.6f}' if isinstance(x,float) else str(x)
 return '\n'.join(['| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |']+['| '+' | '.join(fmt(r.get(k,'')) for k in keys)+' |' for r in rs])
def main():
 audit=json.loads((O/'category_audit.json').read_text(encoding='utf-8'));hist=json.loads((O/'history.json').read_text(encoding='utf-8'));result=json.loads((O/'test_metrics.json').read_text(encoding='utf-8'))
 comparison=[];groups=[];transitions=[];classes=[];known=[]
 for split in ['val','test']:
  paths={'A1':R.parent/'A1/结果'/f'{split}_predictions.csv','A2':O/f'{split}_predictions.csv','A0+A1':R.parent/'A1/融合对照/A0_A1集成'/f'{split}_predictions.csv'}
  data={k:read(v) for k,v in paths.items()};a2=data['A2'];ids=[r['钱币编号'] for r in a2]
  for model,rs in data.items():
   ix={r['钱币编号']:r for r in rs};assert len(ix)==len(ids) and set(ix)==set(ids)
   data[model]=[ix[i] for i in ids];rs=data[model]
   if model=='A0+A1':
    rs=[{**meta,**r,'coin_score':r['ensemble_score'],'pred_label':r['ensemble_pred']} for r,meta in zip(rs,a2)];data[model]=rs
   assert [r['true_label'] for r in rs]==[r['true_label'] for r in a2]
   comparison.append(dict(split=split,model=model,**calc(rs)))
   for g in ['AU','XF','MS','92','97','98']:
    subset=[r for r in rs if (int(r['true_label'])==0 and r['原始评级'].startswith(g))] if g in ['AU','XF','MS'] else [r for r in rs if int(r['true_label'])==1 and g in r['异常代码']]
    err=sum(r['true_label']!=r['pred_label'] for r in subset)
    groups.append(dict(split=split,model=model,group=g,n=len(subset),errors=err,error_rate=err/len(subset) if subset else None))
   for state in ['known','Unknown']:
    subset=[r for r,meta in zip(rs,a2) if (meta['category_id']=='0')==(state=='Unknown')]
    known.append(dict(split=split,model=model,category_status=state,**calc(subset)))
   for name in sorted({r['category_name'] for r in a2}):
    subset=[r for r,meta in zip(rs,a2) if meta['category_name']==name]
    classes.append(dict(split=split,model=model,category=name,**calc(subset)))
  for old,new in zip(data['A1'],a2):
   olderr=old['true_label']!=old['pred_label'];newerr=new['true_label']!=new['pred_label']
   transition='corrected' if olderr and not newerr else 'introduced' if newerr and not olderr else 'both_wrong' if olderr else 'both_correct'
   transitions.append(dict(split=split,coin_id=new['钱币编号'],true_label=new['true_label'],grade=new['原始评级'],code=new['异常代码'],category=new['category_name'],category_id=new['category_id'],A1_score=old['coin_score'],A2_score=new['coin_score'],A1_pred=old['pred_label'],A2_pred=new['pred_label'],transition=transition))
 for name,rs in [('comparison',comparison),('group_comparison',groups),('error_transitions',transitions),('category_comparison',classes),('known_unknown_comparison',known)]:t.write_csv(O/f'{name}.csv',rs)
 ts=[]
 for split in ['val','test']:
  for label in ['0','1']:
   rr=[r for r in transitions if r['split']==split and r['true_label']==label]
   ts.append(dict(split=split,label=label,**{k:sum(r['transition']==k for r in rr) for k in ['corrected','introduced','both_wrong']}))
 t.write_csv(O/'transition_summary.csv',ts)
 # Verify immutable sources, mapping, predictions, and saved checkpoint before completion.
 baseline=json.loads((O/'historical_hashes.json').read_text(encoding='utf-8'))
 changed=[p for p,h in baseline.items() if hashlib.sha256(t.Path(p).read_bytes()).hexdigest()!=h];assert not changed,changed
 cfg=json.loads((t.MODEL/'config.json').read_text(encoding='utf-8'))
 for split,file in t.FILES.items():assert hashlib.sha256((t.SPLIT/file).read_bytes()).hexdigest()==cfg['split_sha256'][split]
 frozen=json.loads((t.MODEL/'frozen_evaluation.json').read_text(encoding='utf-8'))
 assert hashlib.sha256((t.MODEL/'best.pt').read_bytes()).hexdigest()==frozen['checkpoint_sha256']
 assert hashlib.sha256((t.MODEL/'category_mapping.json').read_bytes()).hexdigest()==frozen['mapping_sha256']
 best=t.torch.load(t.MODEL/'best.pt',map_location='cpu',weights_only=False)
 assert best['model']['embedding.weight'].shape==(audit['embedding_rows'],32)
 assert best['model']['head.0.weight'].shape==(256,1568)
 assert all(t.torch.isfinite(v).all() for v in best['model'].values())
 def get(split,model):return next(r for r in comparison if r['split']==split and r['model']==model)
 delta={s:{k:get(s,'A2')[k]-get(s,'A1')[k] for k in ['accuracy','precision','recall','f1','auroc','ap','fp','fn']} for s in ['val','test']}
 t.save_json(O/'metric_deltas.json',delta)
 text='# A2：真实类别引导的双面钱币异常检测\n\n'
 text+='类别输入采用 Dataset V1 原类别的完整年份／面额／细分版别路径。训练集原有61种路径、97种PCGS品种编号；选择已有可核对的分类体系，不将PCGS详情字符串或评级拼接为类别。历史96枚关联未确认全部Unknown，另验证集2枚未见类别映射Unknown。类别字段仅经Embedding输入，不参与辅助损失。\n\n'
 text+=f"有效已知类别 {audit['known_categories']}，加 Unknown 共 {audit['embedding_rows']} 个Embedding行。各集合：`{json.dumps(audit['splits'],ensure_ascii=False)}`。全部样本、二分类标签和划分保持不变。\n\n"
 text+='采用A1相同ImageNet权重、共享ConvNeXt-Tiny、384分辨率及独立±10度旋转。双面1536维+类别32维，经1568→256→1 GELU MLP；全部参数训练。Batch8、AdamW、主干2e-5、MLP与Embedding1e-4、WD0.01、AMP与同批溢出重试；种子42；15–50轮、耐心10、调度完整沿用A1。配置见模型/config.json，未使用A1 best.pt。\n\n'
 text+=f"最佳Epoch {result['best_epoch']}，完成 {len(hist)} 轮，训练/验证/保存累计 {result['training_seconds']/60:.2f} 分钟；阈值 {result['threshold']:.10g}。先按验证AUROC选模型，再按验证F1最大化选阈值（同分取最高阈值），冻结配置后测试一次。报告仅读取已保存预测。\n\n"
 text+='正式训练完成后，最终验证加载器新增进程触发Windows WinError1455（页面文件不足）。保留已完成15轮训练与best/last权重，终止本实验残留进程，单独以workers=0完成最终验证和首次测试；模型、batch、数据预处理、阈值规则均不变。原执行训练代码保存在代码/train_executed.py，恢复记录见结果/evaluation_recovery.json。\n\n'
 text+='## 固定对照指标\n\n'+table(comparison,['split','model','accuracy','precision','recall','f1','auroc','ap','fp','fn'])+'\n\n'
 text+='混淆矩阵行=真实0/1，列=预测0/1：`'+json.dumps(result['confusion_matrix'])+'`。\n\n'
 text+='## 主要分组\n\n'+table(groups,['split','model','group','n','errors','error_rate'])+'\n\n'
 text+='## A1错误迁移\n\n'+table(ts,['split','label','corrected','introduced','both_wrong'])+'\n\n逐币证书编号、概率、类别和错误迁移见结果/error_transitions.csv。标签0为误报，1为漏检。\n\n'
 text+='## 已知类别和Unknown\n\n'+table(known,['split','model','category_status','n','accuracy','f1','auroc','fp','fn'])+'\n\nUnknown样本少，分组AUROC/AP缺少对应标签时留空，不据此作稳定结论。\n\n'
 text+='## 版别变化\n\n'
 changes=[]
 for a in [r for r in classes if r['split']=='test' and r['model']=='A2']:
  b=next(r for r in classes if r['split']=='test' and r['model']=='A1' and r['category']==a['category'])
  changes.append(dict(category=a['category'],n=a['n'],A1_fp=b['fp'],A2_fp=a['fp'],A1_fn=b['fn'],A2_fn=a['fn'],net_errors_reduced=b['fp']+b['fn']-a['fp']-a['fn']))
 text+=table(sorted(changes,key=lambda r:r['net_errors_reduced'],reverse=True),['category','n','A1_fp','A2_fp','A1_fn','A2_fn','net_errors_reduced'])+'\n\n'
 dist=read(O/'category_distribution.csv');train_dist=[r for r in dist if r['split']=='train' and int(r['total'])>0]
 pure=[r for r in train_dist if int(r['normal'])==0 or int(r['abnormal'])==0];small=[r for r in train_dist if int(r['total'])<10]
 text+=f"训练中 {len(pure)} 个类别仅含一种二分类标签，{len(small)} 个类别少于10枚；完整正常/异常计数见category_distribution.csv。Embedding可能利用类别先验，不能把收益全部归因于视觉缺陷识别；没有按分布删样本或改变训练目标。\n\n"
 text+='## 实验判断\n\n'
 text+='相对A1的指标差值：`'+json.dumps(delta,ensure_ascii=False)+'`。\n\n'
 if delta['val']['auroc']>0 and delta['val']['f1']>=0:
  text+='验证集支持将真实类别条件作为后续专家模型的候选组成部分；仍需独立数据和多种子检验。'
 else:text+='验证集尚未同时支持排序和分类效果改善，现阶段不建议将类别分支作为已验证有效的默认组件；可保留为研究对照。'
 text+='本次仅为真实类别输入的单种子固定划分实验，不能外推到预测类别输入；所有测试分组仅作描述，不据此修改方案。\n\n'
 text+='原始划分CSV和A0/A1完整文件哈希复核通过。按用户确认保留SSD共享图像缓存，仅在全部结果核验后清理A2自建临时文件；E盘原始图片保持不变。\n'
 (R/'实验报告.md').write_text(text,encoding='utf-8')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(1,2,figsize=(11,4))
 for ax,split in zip(axs,['val','test']):
  rr=[get(split,m) for m in ['A1','A2','A0+A1']];x=np.arange(3)
  for i,k in enumerate(['auroc','ap','f1']):ax.bar(x+(i-1)*.25,[r[k] for r in rr],.25,label=k)
  ax.set(xticks=x,xticklabels=['A1','A2','A0+A1'],ylim=(0,1),title=split);ax.legend()
 fig.tight_layout();fig.savefig(O/'model_comparison.png',dpi=160);plt.close(fig)
 t.save_json(O/'verification.json',{'passed':True,'historical_files_verified':len(baseline),'split_hashes_unchanged':True,'checkpoint_and_mapping_verified':True,'prediction_alignment_verified':True})
 required=[R/'实验报告.md',t.MODEL/'best.pt',t.MODEL/'last.pt',O/'test_predictions.csv',O/'val_predictions.csv',O/'model_comparison.png',O/'confusion_matrix.png']
 assert all(p.exists() and p.stat().st_size for p in required)
 t.save_json(O/'COMPLETE.json',{'completed':True,'unix':time.time(),'artifacts':{str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in required}})
 print(json.dumps({'completed':True,'delta':delta,'transitions':ts,'best_epoch':result['best_epoch']}),flush=True)
if __name__=='__main__':main()




