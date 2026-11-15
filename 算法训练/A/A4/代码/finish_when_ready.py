import time,subprocess,sys,json
from pathlib import Path
R=Path(__file__).resolve().parents[1];O=R/'结果'
while not all((R/n/'结果/TRAIN_COMPLETE.json').exists() for n in ['A4a','A4b']):time.sleep(10)
commands=[['evaluate.py','freeze'],['evaluate.py','infer'],['evaluate.py','analyze'],['report.py'],['interpret.py'],['verify_final.py']]
with (O/'finalize.log').open('a',encoding='utf-8') as f:
 for command in commands:
  if command==['evaluate.py','freeze'] and (R/'frozen_evaluation.json').exists():continue
  f.write('RUN '+repr(command)+'\n');f.flush()
  p=subprocess.run([sys.executable,str(R/'代码'/command[0]),*command[1:]],stdout=f,stderr=subprocess.STDOUT)
  if p.returncode:
   (O/'FINALIZE_ERROR.json').write_text(json.dumps({'command':command,'returncode':p.returncode}),encoding='utf-8');raise SystemExit(p.returncode)
(O/'PIPELINE_COMPLETE.json').write_text(json.dumps({'completed_unix':time.time(),'note':'Artifacts generated; final narrative and visual review pending'}),encoding='utf-8')
