import os,subprocess,csv,io,time,json,sys
from pathlib import Path
O=Path(__file__).resolve().parents[1]
def choose():
 r=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 candidates=[]
 for row in csv.reader(io.StringIO(r)):
  idx,uid,total,used,util=[x.strip() for x in row]
  if float(used)<800 and float(util)<10: candidates.append((float(total)-float(used),uid))
 return max(candidates)[1] if candidates else None
while not (uid:=choose()):
 (O/'结果/queue.json').write_text(json.dumps({'state':'waiting_for_idle_GPU','time':time.time()})); time.sleep(60)
os.environ['CUDA_VISIBLE_DEVICES']=uid
(O/'结果/gpu_selection.json').write_text(json.dumps({'uuid':uid,'time':time.time(),'policy':'dynamic idle GPU; memory<800MiB and util<10%; one device only'}))
subprocess.run([sys.executable,str(O/'代码'/('smoke.py' if '--smoke' in sys.argv else 'train.py'))]+([] if '--smoke' in sys.argv else ['--mode','train']),check=True)
