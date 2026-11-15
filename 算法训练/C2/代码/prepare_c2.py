import json, shutil, hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1]; C0=R.parent/'C/C0'
def save(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2),'utf8')
for name in ['manifest.json','label_mapping.json','grade_counts.csv','data_audit.json','train_manifest.csv','val_manifest.csv','test_manifest.csv']:
    shutil.copy2(C0/name,R/name)
shutil.copy2(C0/'代码/b_reference.py',R/'代码/b_reference.py')
shutil.copy2(C0/'代码/train.py',R/'代码/c0_train_reference.txt')
s=(C0/'代码/train.py').read_text('utf8')
s=s.replace("R.parent.parent/'B/B0/数据缓存'", "R.parent/'B/B0/数据缓存'")
s=s.replace("name=='C0'", "name=='C2'")
s=s.replace("    counts=np.bincount", "    freeze_info=freeze(m);save(out/'freeze_verification.json',freeze_info)\n    counts=np.bincount")
s=s.replace("save(out/'config.json',cfg)", "cfg.update(freeze=freeze_info,train_metric_protocol='train_macro_f1: original online augmented; train_eval_*: epoch-end eval without augmentation, same fixed 20 labels as validation');save(out/'config.json',cfg)")
s=s.replace("assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())", "assert all((p.grad is not None and torch.isfinite(p.grad).all()) if p.requires_grad else p.grad is None for p in m.parameters())")
s=s.replace("finite_gradients=True,backbone_backward=True", "finite_gradients=True,frozen_gradients_none=True,trainable_gradients_present=True")
s=s.replace("'params':m.net.parameters()", "'params':[p for p in m.net.parameters() if p.requires_grad]")
s=s.replace("    history=[];best=-1.", "    initial_frozen={n:p.detach().cpu().clone() for n,p in m.named_parameters() if not p.requires_grad}\n    initial_trainable={n:p.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad}\n    history=[];best=-1.")
s=s.replace("nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()", "nn.utils.clip_grad_norm_(m.parameters(),5,error_if_nonfinite=True);opt.step()\n            if epoch==1 and offset==0:\n                verify_updates(m,initial_frozen,initial_trainable,out/'first_update_verification.json')")
s=s.replace("        sched.step(f);h=", "        ty,tp,tl=b.predict(m,Coins(train),w);tm=metrics(ty,tp);online=metrics(np.array(yy),np.array(pp));vm=metrics(vy,vp)\n        sched.step(f);h=")
s=s.replace("history.append(h)", "h.update(train_accuracy=online['accuracy'],val_accuracy=vm['accuracy'],train_eval_accuracy=tm['accuracy'],train_eval_macro_f1=tm['macro_f1'],train_eval_loss=tl);history.append(h)")
s=s.replace("gc.collect();torch.cuda.empty_cache();finish(m,out,manifest,w,history,cfg)", "gc.collect();torch.cuda.empty_cache();verify_updates(m,initial_frozen,initial_trainable,out/'final_update_verification.json');finish(m,out,manifest,w,history,cfg)\n    from report import complete_report\n    complete_report(m,out,manifest,w,history,cfg)")
# Keep completion marker for the complete evaluated and reported pipeline only.
s=s.replace("save(out/'COMPLETE.json',dict(best_epoch=ck['epoch'],epochs=len(history),metrics=results))", "save(out/'EVALUATED.json',dict(best_epoch=ck['epoch'],epochs=len(history),metrics=results))")
addition='''
def freeze(m):
    for p in m.net.features[:7].parameters(): p.requires_grad_(False)
    total=sum(p.numel() for p in m.parameters());trainable=sum(p.numel() for p in m.parameters() if p.requires_grad)
    return dict(total_parameters=total,trainable_parameters=trainable,trainable_ratio=trainable/total,frozen_modules='net.features.0..6: stem, stages 1-3 and all downsampling',trainable_modules='net.features.7 stage4, net.classifier.0 final norm, head.0 fusion and head.2 classification',frozen_train_mode='C0 unchanged: stochastic depth remains active in train mode; no BatchNorm',trainable_names=[n for n,p in m.named_parameters() if p.requires_grad])
def verify_updates(m,frozen,trainable,path):
    assert all(p.grad is None for n,p in m.named_parameters() if n in frozen)
    unchanged=all(torch.equal(p.detach().cpu(),frozen[n]) for n,p in m.named_parameters() if n in frozen)
    changed=[n for n,p in m.named_parameters() if n in trainable and not torch.equal(p.detach().cpu(),trainable[n])]
    assert unchanged and len(changed)==len(trainable), (unchanged, set(trainable)-set(changed))
    save(path,dict(frozen_unchanged=unchanged,frozen_gradients_none=True,all_trainable_tensors_updated=True,changed=changed))
'''
s=s.replace("def run(name):",addition+"\ndef run(name):")
(R/'代码/train.py').write_text(s,'utf8')
cfg=json.loads((C0/'config.json').read_text('utf8'));cfg.update(experiment='C2',status='queued',freeze='features.0..6 frozen; features.7 + classifier.0 + head trainable',source_experiment=str(C0))
save(R/'queued_config.json',cfg)
inputs=[R/'manifest.json',R/'label_mapping.json',R.parent/'B/B0/数据缓存/a8.pt',R.parent/'B/B0/数据缓存/a8_val.csv',R.parent/'B/B0/数据缓存/a8_test.csv',R.parent/'B/B0/数据缓存/a8_threshold.json']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
assert hashes[str(inputs[2])]==json.loads((C0/'initialization.json').read_text('utf8'))['sha256']
save(R/'input_hashes.json',hashes)
print('Prepared C2; original A8 hash verified; exact C0 manifests copied.')
