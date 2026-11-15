import subprocess, sys, time, json
from pathlib import Path
R=Path(__file__).resolve().parents[1]
while True:
    log=(R/'结果/train_stdout.log').read_text(encoding='utf8',errors='replace')
    err=(R/'结果/train_stderr.log').read_text(encoding='utf8',errors='replace')
    if 'FINAL ' in log:
        subprocess.run([sys.executable,str(R/'代码/evaluate.py')],check=True)
        break
    if 'Traceback (most recent call last)' in err:
        raise RuntimeError('Training failed; inspect train_stderr.log. Evaluation not run.')
    time.sleep(30)
