"""Read-only host/device and live worker counters; no file content reads."""
import datetime
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

p = Path(__file__).resolve().parent
deadline = datetime.datetime(2026, 9, 14, 9, tzinfo=datetime.timezone.utc).timestamp()
(p / 'storage-io-process.json').write_text(json.dumps({'pid': os.getpid(), 'starttime': Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ', 1)[1].split()[19], 'deadline': deadline}))
code = r'''
import json,subprocess,time,sys
from pathlib import Path
rank=int(sys.argv[1]);stats=list(map(int,Path('/sys/block/nvme0n1/stat').read_text().split()))
rows=subprocess.check_output(['docker','top',f'qwen38-kv-paging-r{rank}','-eo','pid,comm'],text=True).splitlines()[1:]
workers=[]
for row in rows:
 pid,comm=row.split(None,1)
 if 'Worker' not in comm:continue
 f=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
 io={k:int(v) for k,v in (line.split(':') for line in Path(f'/proc/{pid}/io').read_text().splitlines())}
 workers.append({'pid':int(pid),'starttime':f[19],'io':io})
mem={k:int(v.split()[0]) for k,v in (line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines()) if k in ('MemAvailable','Cached','Dirty','Writeback','SwapFree')}
print(json.dumps({'time':time.time(),'rank':rank,'device_read_bytes':stats[2]*512,'device_write_bytes':stats[6]*512,'device_io_ms':stats[9],'workers':workers,'memory_kib':mem}))
'''
with (p / 'storage-io.jsonl').open('a') as output:
    while time.time() < deadline:
        record = {'time': time.time(), 'ranks': [], 'scope': 'Device counters include all host I/O. Worker counters include all worker files; compare deltas only with unchanged PID/starttime. No retrospective physical-byte claim.'}
        for rank in (0, 1):
            cmd = ['sudo', '-n', 'python3', '-c', code, str(rank)]
            if rank:
                cmd = ['ssh', 'jon@192.168.100.11', shlex.join(cmd)]
            try:
                record['ranks'].append(json.loads(subprocess.check_output(cmd, text=True, timeout=20)))
            except Exception as error:
                record['ranks'].append({'rank': rank, 'error': repr(error)})
        output.write(json.dumps(record) + '\n')
        output.flush()
        print(json.dumps({'time': record['time'], 'ranks': record['ranks']}), flush=True)
        time.sleep(min(30, max(0, deadline - time.time())))
