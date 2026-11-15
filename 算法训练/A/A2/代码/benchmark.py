import train as t
import time,json,gc
if __name__=='__main__':
 t.seed();m=t.pretrained();opt=t.optimizer(m);scaler=t.torch.amp.GradScaler('cuda',init_scale=1024)
 ds=t.Coins('train',True);dl=t.loader(ds,8,True)
 t.train_epoch(m,opt,scaler,dl,limit=8)
 t.torch.cuda.reset_peak_memory_stats();r=t.train_epoch(m,opt,scaler,dl,limit=64)
 r['steps']=64;r['estimated_training_epoch_seconds']=len(ds)/r['coins_per_second']
 r['estimated_15_epochs_minutes_training_only']=r['estimated_training_epoch_seconds']*15/60
 r['estimated_50_epochs_minutes_training_only']=r['estimated_training_epoch_seconds']*50/60
 assert m.embedding.weight.grad is None
 assert all(p.requires_grad for p in m.parameters())
 assert m.head[0].in_features==1568
 assert opt.state[m.embedding.weight]['step']>0
 t.save_json(t.OUT/'benchmark.json',r);print(json.dumps(r),flush=True)
