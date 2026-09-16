import time,json,urllib.request
from pathlib import Path
p=Path(__file__).parent;end=json.loads((p/'start.json').read_text())['epoch']+180;key=Path('/home/jon/.config/qwen38/api-key').read_text().strip()
with (p/'metrics-valid.jsonl').open('w') as f:
 while time.time()<end:
  t=time.time();row={'time':t}
  try:
   req=urllib.request.Request('http://127.0.0.1:30001/metrics',headers={'Authorization':'Bearer '+key});raw=urllib.request.urlopen(req,timeout=3).read().decode();m={}
   for line in raw.splitlines():
    if not line.startswith('#') and any(x in line for x in ['num_requests_running{','num_requests_waiting{','generation_tokens_total{','prompt_tokens_total{','kv_offload_store_bytes_total{','kv_offload_load_bytes_total{','num_preemptions_total{','kv_cache_usage_perc{']):
     k,v=line.rsplit(' ',1);k=k.split('{')[0];m[k]=m.get(k,0)+float(v)
   row['metrics']=m;row['proc']={}
   for label,pid in [('worker',2416653),('engine',2416556)]:row['proc'][label]={k:Path(f'/proc/{pid}/{k}').read_text() for k in ['stat','wchan','io']}
  except Exception as e:row['error']=str(e)
  f.write(json.dumps(row)+'\n');f.flush();time.sleep(max(0,.5-(time.time()-t)))
