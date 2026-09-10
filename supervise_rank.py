#!/usr/bin/env python3
"""Supervise the attached rank and stop it on sustained host memory pressure."""
import json
from runtime_config import load_config, model_snapshot, resolve_path
import signal
import subprocess
import sys
import time
from pathlib import Path

root=Path(__file__).resolve().parent
rank=int(sys.argv[1]);name=f'qwen38-kv-paging-r{rank}'
cfg=load_config(root)
from paging_ops import prepare_disk, verify_runtime
verify_runtime(cfg, root, rank)
prepare_disk(cfg, root, rank)
stopping=False
def stop(signum=None,frame=None):
    global stopping
    stopping=True
signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)
proc=subprocess.Popen([sys.executable,str(root/'launch_rank.py'),str(rank)])
low=0;iteration=0;failed=False
try:
    while proc.poll() is None and not stopping:
        mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
        available=mem['MemAvailable']/2**30;free=mem['MemFree']/2**30
        low=low+1 if available<8 or (free<2 and available<10) else 0
        if iteration%10==0 or low:
            print(json.dumps({'time':time.time(),'rank':rank,'available_gib':round(available,2),'free_gib':round(free,2),
                              'swap_used_gib':round((mem['SwapTotal']-mem['SwapFree'])/2**30,3),'low_samples':low}),flush=True)
        if low>=5:
            print('Stopping rank: sustained unified-memory pressure',flush=True)
            failed=True;break
        if rank==0 and iteration%15==0:
            status=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',cfg['worker_ip'],
                                   'systemctl is-active qwen38-next-qwen-fp8-worker.service'],capture_output=True,text=True,timeout=10)
            if status.stdout.strip() not in ('active','activating'):
                print('Stopping head: worker service is not active',flush=True)
                failed=True;break
        iteration+=1;time.sleep(1)
finally:
    subprocess.run(['docker','stop','-t','45',name],timeout=60)
    try:proc.wait(timeout=15)
    except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=10)
sys.exit(1 if failed else (0 if stopping else proc.returncode))
