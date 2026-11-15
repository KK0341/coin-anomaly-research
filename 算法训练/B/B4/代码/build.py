from pathlib import Path
p=Path(__file__).parent
s=(p/'b0_reference.py').read_text('utf8')
start=s.index('def finish(');end=s.index('\ndef run(',start)
s=s[:start]+'''def finish(m,out,manifest,w,history,cfg):
    save(out/'validation_selection.json',dict(best_epoch=torch.load(out/'best.pt',map_location='cpu',weights_only=False)['epoch'],test_policy='checkpoint and configuration frozen before one final test'))
    from evaluate import finish_all
    finish_all(m,out,manifest,w,history,cfg)
''' + s[end:]
s=s.replace("if name=='B0'", "if name=='B4'").replace("choices=['B0','B1']","choices=['B4']")
s=s.replace('    def forward(self,x): return self.head(self.net(x.flatten(0,1)).reshape(-1,1536))', '''        g=torch.Generator().manual_seed(42004)
        self.centers=nn.Parameter(nn.functional.normalize(torch.randn(6,256,generator=g),dim=1))
    def forward(self,x,features=False):
        h=self.head[1](self.head[0](self.net(x.flatten(0,1)).reshape(-1,1536)))
        z=self.head[2](h)
        return (z,h) if features else z
    def center_distances(self,h,y):
        return (nn.functional.normalize(h.float(),dim=1)-nn.functional.normalize(self.centers.float(),dim=1)[y]).square().sum(1)''')
s=s.replace("{'head.2.weight','head.2.bias'}", "{'head.2.weight','head.2.bias','centers'}")
s=s.replace("{'params':m.head.parameters(),'lr':1e-4}]", "{'params':m.head.parameters(),'lr':1e-4},{'params':[m.centers],'lr':1e-4,'weight_decay':0.}]")
s=s.replace('min_lr=[1e-7,1e-6]', 'min_lr=[1e-7,1e-6,1e-6]')
s=s.replace("cfg.update(execution_host=", "cfg.update(center_weight=.1,center_location='post-GELU 256-dimensional hidden features',center_loss='mean squared Euclidean distance of L2 normalized features and target centers; float32',center_optimizer='AdamW lr1e-4 weight_decay0; same plateau schedule min_lr1e-6',center_initialization='independent generator seed42004, unit random centers; does not alter B0 network RNG',weight_decay=.01,workers=4,gradient_clip_norm=5,execution_host=")
s=s.replace('ds=Coins(trainrows,True); val=Coins(valrows)', "ds=Coins(trainrows,True); val=Coins(valrows); initial_centers=nn.functional.normalize(m.centers.detach(),dim=1).clone()\n    assert len(trainrows)==1583 and len(valrows)==198\n    assert '5060' in torch.cuda.get_device_name()")
s=s.replace('it=iter(dl)', 'it=iter(dl); center_sum=0.; center_counts=np.zeros(6); center_by_class=np.zeros(6); grad_by_class=np.zeros(6); before_centers=m.centers.detach().clone()')
s=s.replace('opt.zero_grad(set_to_none=True)', 'group_n=sum(len(y) for _,y,_ in batches); opt.zero_grad(set_to_none=True)')
s=s.replace("with torch.autocast('cuda',dtype=torch.bfloat16): z=m(x.cuda(non_blocking=True))\n                loss=", "with torch.autocast('cuda',dtype=torch.bfloat16): z,hfeat=m(x.cuda(non_blocking=True),features=True)\n                cl=m.center_distances(hfeat,y); center_sum+=cl.detach().sum().item()\n                for k in range(6):\n                    mask=y==k; center_counts[k]+=mask.sum().item(); center_by_class[k]+=cl.detach()[mask].sum().item()\n                loss=")
s=s.replace('(loss/group_den).backward()', '(loss/group_den+.1*cl.sum()/group_n).backward()')
s=s.replace('            nn.utils.clip_grad_norm_', '            assert m.centers.grad is not None and torch.isfinite(m.centers.grad).all()\n            grad_by_class+=m.centers.grad.detach().norm(dim=1).cpu().numpy()\n            nn.utils.clip_grad_norm_')
s=s.replace("history.append(h)", "h.update(train_ce=total/denom,train_center_loss=center_sum/len(ds),train_total_loss=total/denom+.1*center_sum/len(ds),center_lr=opt.param_groups[2]['lr'],peak_memory_MiB=torch.cuda.max_memory_allocated()/2**20); history.append(h)\n        cn=nn.functional.normalize(m.centers.detach(),dim=1)\n        center_log=dict(epoch=epoch,codes=CODES,class_counts=center_counts.tolist(),class_center_loss=(center_by_class/center_counts).tolist(),summed_gradient_norm=grad_by_class.tolist(),parameter_update_norm=(m.centers.detach()-before_centers).norm(dim=1).cpu().tolist(),center_norms=m.centers.detach().norm(dim=1).cpu().tolist(),cosine_similarity=(cn@cn.T).cpu().tolist(),distance_from_initial=(cn-initial_centers).norm(dim=1).cpu().tolist())\n        with (out/'center_history.jsonl').open('a',encoding='utf8') as cf: cf.write(json.dumps(center_log)+'\\n')")
s=s.replace('m.eval(); ys=[]; ps=[]; total=0.; denom=0.', 'm.eval(); ys=[]; ps=[]; total=0.; denom=0.; center_eval_sum=0.; center_eval_n=0')
s=s.replace("with torch.autocast('cuda',dtype=torch.bfloat16): z=m(x.cuda(non_blocking=True))", "with torch.autocast('cuda',dtype=torch.bfloat16): z,h=m(x.cuda(non_blocking=True),features=True)\n        center_eval_sum+=m.center_distances(h,y.cuda()).sum().item(); center_eval_n+=len(y)")
s=s.replace('return np.array(ys),np.array(ps),total/denom if denom else 0.', 'm.last_val_center_loss=center_eval_sum/center_eval_n\n    return np.array(ys),np.array(ps),total/denom if denom else 0.')
s=s.replace('h.update(train_ce=', 'h.update(val_center_loss=m.last_val_center_loss,val_total_loss=vl+.1*m.last_val_center_loss,train_ce=')
(p/'train.py').write_text(s,'utf8')
