from pathlib import Path
import json,hashlib
R=Path('E:/异常检测/算法训练'); O=R/'A10_预训练骨干对照实验'
s=(R/'A8_高分辨率实验/代码/train.py').read_text(encoding='utf8')
s=s.replace("FILES={'train':'训练集.csv','val':'验证集.csv','test':'测试集.csv'}","FILES={'train':'训练集.csv','val':'验证集.csv'}")
s=s.replace('T.Resize((512,512)','T.Resize((504,504)')
a=s.index('class Dual(nn.Module):'); b=s.index('\ndef optimizer(m):',a)
s=s[:a]+'''class Dual(nn.Module):
    def __init__(self):
        super().__init__()
        source=json.loads((ROOT/'来源/provenance.json').read_text(encoding='utf8'))
        self.net=torch.hub.load(source['source_dir'],'dinov2_vits14',source='local',pretrained=False)
        self.head=nn.Sequential(nn.Linear(768,256),nn.GELU(),nn.Linear(256,1))
    def forward(self,x):
        features=self.net(x.flatten(0,1)).reshape(-1,768)
        z=self.head(features).squeeze(1)
        return z,z[:,None]

def pretrained():
    source=json.loads((ROOT/'来源/provenance.json').read_text(encoding='utf8'))
    p=MODEL/'dinov2_vits14_pretrain.pth'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256']
    state=torch.load(p,map_location='cpu',weights_only=True)
    m=Dual(); result=m.net.load_state_dict(state,strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    assert all(torch.equal(m.net.state_dict()[k],v) for k,v in state.items())
    assert all(p.requires_grad for p in m.net.parameters())
    save_json(OUT/'weight_loading.json',dict(strict=True,keys=len(state),all_tensors_equal=True,
        backbone_parameters=sum(p.numel() for p in m.net.parameters()),total_parameters=sum(p.numel() for p in m.parameters()),
        feature_per_side=384,concat=768,all_backbone_requires_grad=True,sha256=source['sha256']))
    return m.cuda()

''' +s[b:]
a=s.index('def finish(m,config,history):'); b=s.index('\ndef train():',a)
s=s[:a]+'''def finish(m,config,history):
    best=torch.load(MODEL/'best.pt',map_location='cpu',weights_only=False); m.load_state_dict(best['model'])
    val=Coins('val'); values=predict(m,loader(val,config['batch_coins'],workers=0))
    precision,recall,thresholds=precision_recall_curve(values[0],values[1]); fs=2*precision[:-1]*recall[:-1]/(precision[:-1]+recall[:-1]+1e-12)
    threshold=float(thresholds[np.flatnonzero(fs==fs.max())[-1]])
    save_json(MODEL/'threshold.json',{'threshold':threshold,'source':'validation only','best_epoch':best['epoch'],'val_metrics':metrics(values[0],values[1],threshold)})
    write_predictions('val',val,values,threshold)
    save_json(OUT/'TRAIN_COMPLETE.json',dict(epochs=len(history),best_epoch=best['epoch'],training_seconds=sum(h['epoch_total_seconds'] for h in history),peak_allocated_MiB=max(h['allocated_MiB'] for h in history)))
    import evaluate
    evaluate.main()

''' +s[b:]
s=s.replace("vl=loader(val,config['batch_coins'])","vl=loader(val,config['batch_coins'],workers=0)")
(O/'代码/train.py').write_text(s,encoding='utf8')
c=json.loads((R/'A8_高分辨率实验/模型/config.json').read_text(encoding='utf8'))
for k in ['training_start_unix_estimate','timing_start_method']: c.pop(k,None)
c.update(experiment='A10b',batch_coins=2,images_per_batch=4,accumulation=4,effective_batch_coins=8,resolution=504,workers=2,weights='official DINOv2 ViT-S/14 no registers',architecture='shared ViT-S/14 normalized CLS 384 per side -> concat768 -> Linear256 GELU Linear1',initialization='strict official complete backbone load',augmentation='resize504, same A8 rotation and ImageNet normalization',head_parameters='MLP; entire backbone trainable; mask token unused without masking',validation_workers=0)
c['split_sha256'].pop('test'); c['loader']['workers']=2
(O/'模型/config.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf8'); (O/'结果/loader_selection.json').write_text(json.dumps(c['loader'],ensure_ascii=False,indent=2),encoding='utf8')
protocol={'variant':'A10b','seed':42,'train_full_backbone':True,'resolution':504,'effective_batch':8,'selection':'validation AUROC','max_epochs':50,'min_epochs':15,'patience':10,'fusion_gate':'AUROC gain >=0.005 OR exact TP154 FP reduction >=5 OR 92 FN reduction >=3 at exact TP154','fusion':'if gate passed: same alpha0.2 A0 +0.8 new; then same old A5 CDF knots and 0.1 A5b weight; no A9 or reason heads','test':False}
(O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
print('prepared')
