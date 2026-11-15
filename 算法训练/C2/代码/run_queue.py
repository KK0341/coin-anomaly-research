"""Local, fail-closed queue. Never terminates or changes another task."""
import os,sys,json,time,subprocess,msvcrt,ctypes,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parents[1]
def status(state,**kw):
    p=R/'queue_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(state=state,queue_pid=os.getpid(),time=time.time(),**kw),ensure_ascii=False,indent=2),'utf8');os.replace(tmp,p)
def busy():
    xml=subprocess.check_output(['nvidia-smi','-q','-x'],text=True)
    gpu=ET.fromstring(xml).find('gpu');assert gpu is not None
    processes=[]
    for p in gpu.findall('.//process_info'):
        kind=p.findtext('type','');name=p.findtext('process_name','')
        desktop=kind=='C+G' and name.lower().endswith('\\chatgpt.exe') and 'openai.codex_' in name.lower()
        if 'C' in kind and not desktop:processes.append(dict(pid=p.findtext('pid'),name=name,type=kind))
    util=int(gpu.findtext('utilization/gpu_util').split()[0]);memory=int(gpu.findtext('fb_memory_usage/used').split()[0])
    command="Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    raw=subprocess.check_output(['powershell','-NoProfile','-Command',command],encoding='utf8',errors='replace');rows=json.loads(raw or '[]');rows=rows if isinstance(rows,list) else [rows]
    active=[p for p in rows if p['ProcessId']!=os.getpid() and str(R).lower() not in (p.get('CommandLine') or '').lower() and any(s in (p.get('CommandLine') or '').lower() for s in ['train.py','run_local.py','run_queue.py','gpu_queue.py'])]
    return dict(compute_processes=processes,other_training_or_queues=active,gpu_util=util,memory_mib=memory),not processes and not active and util<=10 and memory<2000
def main():
    own=(R/'queue.lock').open('a+b');own.seek(0);own.write(b'1');own.flush();own.seek(0);msvcrt.locking(own.fileno(),msvcrt.LK_NBLCK,1)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        idle=0
        while True:
            try:
                info,free=busy();idle=idle+1 if free else 0;status('waiting_gpu',consecutive_idle_checks=idle,**info)
                if idle>=3:break
            except Exception as e:idle=0;status('waiting_check_retry',error=repr(e))
            time.sleep(20)
        # Preserve the same shared lock path used by the historical local queue.
        lock=(R.parent/'B/local_runner.lock').open('a+b');lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0)
        while True:
            try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);break
            except OSError:status('waiting_shared_lock');time.sleep(20)
        while True:
            info,free=busy()
            if free:break
            status('waiting_gpu',**info);time.sleep(20)
        for p,digest in json.loads((R/'input_hashes.json').read_text('utf8')).items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,p
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',PYTHONUTF8='1',OMP_NUM_THREADS='4',COIN_DATA_DIR=str(R.parent/'B/B0/数据缓存'))
        started=time.time()
        with (R/'train.log').open('a',encoding='utf8') as log:
            p=subprocess.Popen([sys.executable,'-u',str(R/'代码/train.py'),'C2'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
            status('training',training_pid=p.pid,started=started);rc=p.wait()
        if rc or not (R/'COMPLETE.json').exists():raise RuntimeError(f'C2 failed (exit={rc}); inspect train.log')
        status('complete',wall_seconds=time.time()-started)
    except Exception as e:status('failed',error=repr(e));raise
    finally:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
if __name__=='__main__':main()
