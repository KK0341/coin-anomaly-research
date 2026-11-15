from pathlib import Path
import subprocess,sys,time,json,traceback
root=Path(__file__).resolve().parents[1]
def run(name,*args):
    subprocess.run([sys.executable,'-u',str(root/'代码'/name),*args],check=True)
def main():
    while not (root/'结果/loader_selection.json').exists():time.sleep(5)
    # The selection file is written before final benchmark JSON; wait for completion marker.
    while 'speedup' not in json.loads((root/'结果/io_benchmark.json').read_text(encoding='utf-8')):time.sleep(2)
    b=json.loads((root/'结果/io_benchmark.json').read_text(encoding='utf-8'))
    sec=b['estimated_train_epoch_seconds']
    print(f'PRETRAIN ESTIMATE training_only_epoch={sec:.1f}s; 15_epochs={sec*15/60:.1f}min; 50_epochs={sec*50/60:.1f}min; validation/save overhead extra',flush=True)
    run('fusion.py')
    run('train.py','--mode','train')
    run('report.py')
    (root/'结果/COMPLETE.json').write_text(json.dumps({'completed':True,'unix':time.time()}),encoding='utf-8')
if __name__=='__main__':
    try:main()
    except Exception:
        (root/'结果/FAILED.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
