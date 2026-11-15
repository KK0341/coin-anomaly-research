import os,sys,time,json,subprocess,ctypes,xml.etree.ElementTree as ET,msvcrt
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def status(state,**kw):
    p=R/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(state=state,time=time.time(),**kw),ensure_ascii=False,indent=2),'utf8');os.replace(tmp,p)
def main():
    own=(R/'运行记录/runner.lock').open('a+b');own.seek(0);own.write(b'1');own.flush();own.seek(0);msvcrt.locking(own.fileno(),msvcrt.LK_NBLCK,1)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        idle=0
        while idle<3:
            gpu=ET.fromstring(subprocess.check_output(['nvidia-smi','-q','-x'],text=True)).find('gpu')
            processes=[dict(pid=p.findtext('pid'),name=p.findtext('process_name')) for p in gpu.findall('.//process_info') if p.findtext('type')=='C']
            legacy=R.parent/'local_runner.lock';free=True
            if legacy.exists():
                try:
                    with legacy.open('r+b') as f:msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
                except OSError:free=False
            idle=idle+1 if not processes and free else 0;status('waiting_local_gpu',processes=processes,legacy_lock_free=free,idle_checks=idle)
            if idle<3:time.sleep(20)
        if legacy.exists():legacy.rename(R/'运行记录/legacy_local_runner.lock')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONUTF8='1',COIN_DATA_DIR=str(R.parent/'B0/数据缓存'),OMP_NUM_THREADS='4')
        with (R/'train.log').open('a',encoding='utf8') as log:
            p=subprocess.Popen([sys.executable,'-u',str(R/'代码/train.py'),'B4'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT);status('training',pid=p.pid);rc=p.wait()
        status('complete' if rc==0 else 'failed',returncode=rc)
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
if __name__=='__main__':main()
