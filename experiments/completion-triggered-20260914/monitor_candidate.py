import concurrent.futures,json,subprocess,time,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent
key=Path('/home/jon/.config/qwen38/api-key').read_text().strip()
query=['nvidia-smi','--query-gpu=power.draw,utilization.gpu,clocks.sm,temperature.gpu,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_thermal_slowdown','--format=csv,noheader,nounits']
def gpu(remote):
 return subprocess.check_output((['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','jon@192.168.100.11'] if remote else [])+query,text=True,timeout=10).strip()
def metrics():
 r=urllib.request.Request('http://127.0.0.1:30001/metrics',headers={'Authorization':'Bearer '+key})
 s=urllib.request.urlopen(r,timeout=10).read().decode()
 names=['generation_tokens_total','num_requests_running','num_requests_waiting','kv_offload_load_bytes_total','kv_offload_load_time_total','prefix_cache_queries_total','prefix_cache_hits_total','external_prefix_cache_queries_total','external_prefix_cache_hits_total','prompt_tokens_total','prompt_tokens_cached_total','num_preemptions_total','spec_decode_num_drafts_total','spec_decode_num_accepted_tokens_total']
 return {n:sum(float(l.rsplit(' ',1)[1]) for l in s.splitlines() if l.startswith('vllm:'+n+'{')) for n in names}
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for i in range(37):
  start=time.time();fs=[pool.submit(gpu,False),pool.submit(gpu,True),pool.submit(metrics)]
  rows.append({'time':start,'r0':fs[0].result(),'r1':fs[1].result(),'metrics':fs[2].result()})
  if i<36:time.sleep(max(0,5-(time.time()-start)))
(p/'candidate-performance.json').write_text(json.dumps(rows,indent=2))
a,b=rows[0],rows[-1];dt=b['time']-a['time']
print(json.dumps({'seconds':dt,'aggregate_tps':(b['metrics']['generation_tokens_total']-a['metrics']['generation_tokens_total'])/dt},indent=2))
