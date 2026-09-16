"""Bounded live observation; no serving mutations or inference requests."""
import argparse
import json
import os
import signal
from pathlib import Path
import subprocess
import threading
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--rank', type=int, required=True)
p.add_argument('--worker', type=int, required=True)
p.add_argument('--ple', type=int, required=True)
p.add_argument('--engine', type=int)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--duration', type=int, default=210)
p.add_argument('--py-spy', required=True)
p.add_argument('--profile-delay', type=int, default=40)
p.add_argument('--profile-duration', type=int, default=150)
p.add_argument('--profile-rate', type=int, default=20)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
start = time.time()
offset = time.time_ns() - time.monotonic_ns()
(a.output/'start.json').write_text(json.dumps({**vars(a), 'output':str(a.output),
    'wall_start':start, 'wall_minus_monotonic_ns':offset}, indent=2))
children = []
handles = []
def launch(command, filename):
    f = (a.output/filename).open('w'); handles.append(f)
    proc = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT)
    children.append(proc)
    return proc

sensor = subprocess.Popen(['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu',
    '--format=csv,noheader,nounits','-lms','200'], stdout=subprocess.PIPE, stderr=subprocess.PIPE,text=True)
children.append(sensor)
def sensor_rows():
    last_dump=0
    def dump(at,util):
        for label,pid in pids.items():
            begin=time.time()
            try:
                r=subprocess.run([a.py_spy,'dump','--pid',str(pid),'--nonblocking','--json'],capture_output=True,timeout=5)
                record={'trigger_time':at,'util':util,'start':begin,'end':time.time(),
                    'label':label,'code':r.returncode,'stdout':r.stdout.decode(errors='replace'),
                    'stderr':r.stderr.decode(errors='replace')}
            except subprocess.TimeoutExpired:
                record={'trigger_time':at,'util':util,'start':begin,'end':time.time(),'label':label,'error':'dump timeout'}
            with (a.output/'dip-stacks.jsonl').open('a') as df:df.write(json.dumps(record)+'\n')
    with (a.output/'gpu.jsonl').open('w') as f:
        for line in sensor.stdout:
            now=time.time()
            f.write(json.dumps({'time':now,'values':line.strip()})+'\n');f.flush()
            try:util=float(line.split(',')[0])
            except ValueError:continue
            if util<60 and now-last_dump>5:
                last_dump=now
                threading.Thread(target=dump,args=(now,util),daemon=True).start()
threading.Thread(target=sensor_rows,daemon=True).start()
launch(['docker','logs','--timestamps','--follow','--since','2s',f'qwen38-kv-paging-r{a.rank}'],'server.log')
pids={'worker':a.worker,'ple':a.ple}
if a.engine:pids['engine']=a.engine
profiled=False
try:
    with (a.output/'host.jsonl').open('w') as f, (a.output/'metrics.jsonl').open('w') as mf:
        while time.time()-start < a.duration:
            now=time.time();row={'time':now,'proc':{}}
            if not profiled and now-start>=a.profile_delay:
                profiled=True
                for label,pid in pids.items():
                    launch([a.py_spy,'record','--pid',str(pid),'--rate',str(a.profile_rate),'--duration',str(a.profile_duration),
                        '--format','speedscope','--idle','--nonblocking','--output',str(a.output/(label+'.speedscope.json'))],label+'-profile.log')
                launch(['timeout','--signal=INT',str(a.profile_duration+5),'bpftrace','-q',str(a.output/'ple.bt')],'ple.txt')
                (a.output/'profile-start.json').write_text(json.dumps({'time':time.time()}))
            for label,pid in pids.items():
                try:
                    base=Path('/proc')/str(pid)
                    fields=(base/'stat').read_text().rsplit(')',1)[1].split()
                    row['proc'][label]={'pid':pid,'state':fields[0], 'minflt':int(fields[7]),'majflt':int(fields[9]),
                        'utime':int(fields[11]),'stime':int(fields[12]),'wchan':(base/'wchan').read_text(),
                        'io':(base/'io').read_text(),'schedstat':(base/'schedstat').read_text()}
                except (OSError,ValueError) as e:row['proc'][label]={'error':str(e)}
            for name,path in [('diskstats','/proc/diskstats'),('meminfo','/proc/meminfo'),
                    ('pressure_io','/proc/pressure/io'),('pressure_memory','/proc/pressure/memory'),('pressure_cpu','/proc/pressure/cpu')]:
                row[name]=Path(path).read_text()
            f.write(json.dumps(row)+'\n');f.flush()
            if a.rank==0:
                try:
                    with urllib.request.urlopen('http://127.0.0.1:30001/metrics',timeout=2) as response:raw=response.read().decode()
                    mf.write(json.dumps({'time':now,'raw':raw})+'\n');mf.flush()
                except Exception as e:mf.write(json.dumps({'time':now,'error':str(e)})+'\n');mf.flush()
            time.sleep(max(0,.5-(time.time()-now)))
finally:
    for proc in children:
        if proc.poll() is None:proc.send_signal(signal.SIGINT)
    for proc in children:
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()
    for f in handles:f.close()
    (a.output/'end.json').write_text(json.dumps({'time':time.time(),'returncodes':[p.returncode for p in children]}))
