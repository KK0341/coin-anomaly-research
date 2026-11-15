import subprocess,json,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
for name in ['A4a','A4b']:
 d=R/name
 if (d/'结果/TRAIN_COMPLETE.json').exists():continue
 with (d/'结果/train.log').open('a',encoding='utf-8') as f:
  p=subprocess.run([sys.executable,str(d/'代码/train.py'),'--mode','train'],stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise SystemExit(p.returncode)
print('BOTH_TRAIN_COMPLETE')
