from pathlib import Path
import subprocess,sys,traceback,json,time
R=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 try:
  subprocess.run([sys.executable,'-u',str(R/'代码/train.py'),'--mode','train'],check=True)
  subprocess.run([sys.executable,'-u',str(R/'代码/report.py')],check=True)
 except Exception:
  (R/'结果/FAILED.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
