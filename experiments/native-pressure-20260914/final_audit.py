import json
import re
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path

p=Path(__file__).resolve().parent
start=json.loads((p/'start.json').read_text())['time']
def run(rank,args):
    return subprocess.check_output(['ssh','jon@192.168.100.11',shlex.join(args)] if rank else args,text=True,timeout=30)
def metrics():
    with urllib.request.urlopen('http://127.0.0.1:30001/metrics',timeout=10) as f:raw=f.read().decode()
    names=['kv_cache_usage_perc','kv_completion_resident_blocks','num_requests_running','num_requests_waiting',
        'generation_tokens_total','prefix_cache_hits_total','prefix_cache_queries_total','external_prefix_cache_hits_total',
        'external_prefix_cache_queries_total','kv_offload_store_bytes_total','kv_offload_load_bytes_total','num_preemptions_total']
    return raw,{n:sum(float(x) for x in re.findall(r'^vllm:'+n+r'(?:\{[^\n]*\})? ([\d.eE+-]+)$',raw,re.M)) for n in names}
samples=[]
for i in range(7):
    raw,m=metrics();samples.append({'time':time.time(),'metrics':m})
    if i<6:time.sleep(5)
(p/'final-metrics.txt').write_text(raw)
first,last=samples[0],samples[-1]
result={'time':time.time(),'samples':samples,'aggregate_tps':(last['metrics']['generation_tokens_total']-first['metrics']['generation_tokens_total'])/(last['time']-first['time']),
    'mean_running':sum(s['metrics']['num_requests_running'] for s in samples)/len(samples),
    'scope':'30 second ordinary post-probe traffic; not a matched historical performance comparison.','ranks':{}}
for rank in (0,1):
    cfg=json.loads(run(rank,['cat',str(p.parents[1]/'deploy_config.json')]))
    assert cfg==json.loads((p/f'candidate-r{rank}.json').read_text())
    args=['docker','logs','--since',str(int(start)),f'qwen38-kv-paging-r{rank}']
    r=subprocess.run(['ssh','jon@192.168.100.11',shlex.join(args)] if rank else args,capture_output=True,text=True,timeout=30)
    logs=r.stdout+r.stderr;(p/f'final-r{rank}.log').write_text(logs)
    errors=[line for line in logs.splitlines() if re.search(r'\bERROR\b|Traceback \(most recent|EngineDeadError|OutOfMemoryError|CUDA error|Native prefix eviction save timed out',line)]
    result['ranks'][str(rank)]={'config_verified':True,'error_lines':errors,
        'eviction_save_events':logs.count('GB10_NATIVE_EVICTION save'),
        'completion_snapshot_events':logs.count('Completion snapshot plan')}
with urllib.request.urlopen('http://127.0.0.1:30001/health',timeout=10) as f:result['health']=f.status
result['service']=subprocess.check_output(['systemctl','is-active','qwen38-next-qwen-fp8.service'],text=True).strip()
result['passed']=result['health']==200 and result['service']=='active' and all(not r['error_lines'] for r in result['ranks'].values())
(p/'FINAL-AUDIT.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2),flush=True)
assert result['passed']
