import json
from pathlib import Path
import train
R=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    train.LOADER_OPTIONS['workers']=0
    cfg=json.loads((R/'模型/config.json').read_text(encoding='utf8'))
    history=json.loads((R/'结果/history.json').read_text(encoding='utf8'))
    assert len(history)==15 and max(history,key=lambda h:h['val_auroc'])['epoch']==4
    train.save_json(R/'审计/inference_recovery.json',dict(reason='Windows commit/pagefile exhausted creating additional evaluation workers after full training',change='inference num_workers 4 -> 0 only',training_changed=False,retrain=False))
    train.finish(train.Dual().cuda(),cfg,history)
