import train as t
import json,time
if __name__=='__main__':
 cfg=json.loads((t.MODEL/'config.json').read_text(encoding='utf-8'))
 hist=json.loads((t.OUT/'history.json').read_text(encoding='utf-8'))
 assert len(hist)==15 and not (t.OUT/'test_predictions.csv').exists()
 t.seed();m=t.Dual().cuda()
 t.save_json(t.OUT/'evaluation_recovery.json',{'reason':'Windows WinError1455 while spawning final validation workers after training; test had not started','evaluation_workers':0,'training_unchanged':True,'checkpoint':'best.pt','unix':time.time()})
 t.finish(m,cfg,hist)
 import report
 report.main()
