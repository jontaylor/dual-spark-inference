import subprocess,time,json,threading,urllib.request
from pathlib import Path
p=Path(__file__).parent;start=time.time();duration=180
(p/'start.json').write_text(json.dumps({'epoch':start,'duration':duration}))
processes=[]
for rank,cmd in [(0,[]),(1,['ssh','jon@192.168.100.11'])]:
 args=['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu','--format=csv,noheader,nounits','-lms','200']
 f=(p/f'gpu-r{rank}.csv').open('w');processes.append((subprocess.Popen(cmd+args,stdout=f,stderr=subprocess.STDOUT),f))
for label,pid in [('worker',2416653),('engine',2416556)]:
 f=(p/f'{label}-profile.log').open('w');cmd=['sudo','-n','/home/jon/.local/bin/py-spy','record','--pid',str(pid),'--rate','25','--duration',str(duration),'--format','speedscope','--idle','--nonblocking','--output',str(p/f'{label}.speedscope.json')]
 (p/f'{label}-start.json').write_text(json.dumps({'epoch':time.time(),'pid':pid,'cmd':cmd}));processes.append((subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT),f))
key=Path('/home/jon/.config/qwen38/api-key').read_text().strip()
with (p/'metrics.jsonl').open('w') as f:
 while time.time()-start<duration:
  t=time.time()
  try:
   req=urllib.request.Request('http://127.0.0.1:30001/metrics',headers={'Authorization':'Bearer '+key})
   raw=urllib.request.urlopen(req,timeout=3).read().decode();m={}
   for line in raw.splitlines():
    if line.startswith('#'):continue
    if any(x in line for x in ['num_requests_running{','num_requests_waiting{','generation_tokens_total{','prompt_tokens_total{','kv_offload_store_bytes_total{','kv_offload_load_bytes_total{','num_preemptions_total{','kv_cache_usage_perc{']):
     k,v=line.rsplit(' ',1);k=k.split('{')[0];m[k]=m.get(k,0)+float(v)
   row={'time':t,'metrics':m,'proc':{}}
   for label,pid in [('worker',2416653),('engine',2416556)]:
    row['proc'][label]={k:Path(f'/proc/{pid}/{k}').read_text() for k in ['stat','wchan','io']}
   f.write(json.dumps(row)+'\n');f.flush()
  except Exception as e:f.write(json.dumps({'time':t,'error':str(e)})+'\n');f.flush()
  time.sleep(max(0,0.5-(time.time()-t)))
for proc,f in processes:
 try:proc.wait(timeout=2)
 except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=5)
 f.close()
print('capture complete')
