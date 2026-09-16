import concurrent.futures
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

p=Path(__file__).resolve().parent
out=p/sys.argv[1] if len(sys.argv)>1 else p/('probe-'+str(time.time_ns()))
resume=out.exists()
out.mkdir(exist_ok=resume)
key=Path('/home/jon/.config/qwen38/api-key').read_text().strip()
base='http://127.0.0.1:30001'

def metrics():
    with urllib.request.urlopen(base+'/metrics',timeout=10) as f:s=f.read().decode()
    names=['kv_completion_resident_blocks','kv_offload_store_bytes_total','kv_offload_load_bytes_total',
        'prefix_cache_queries_total','prefix_cache_hits_total','external_prefix_cache_hits_total',
        'kv_cache_usage_perc','num_requests_running','generation_tokens_total','num_preemptions_total']
    return {n:sum(float(x) for x in re.findall(r'^vllm:'+n+r'(?:\{[^\n]*\})? ([\d.eE+-]+)$',s,re.M)) for n in names}

def req(name,payload):
    payload=dict(payload,request_id='nativeprobe-'+out.name+'-'+name)
    r=urllib.request.Request(base+'/v1/completions',data=json.dumps(payload).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    start=time.time()
    with urllib.request.urlopen(r,timeout=600) as f:response=json.load(f)
    (out/(name+'.json')).write_text(json.dumps({'start':start,'end':time.time(),'request':payload,'response':response}))
    print(name,response['usage'],flush=True)
    return response

before=json.loads((out/'before.json').read_text()) if resume else metrics()
(out/'before.json').write_text(json.dumps(before))
source=p.parent/'aggregate-throughput-20260914/after-J/first.json'
prompt=json.loads(source.read_text())['response']['prompt_token_ids']
options=dict(model='qwen3.8-flash-next',temperature=0,seed=917352,ignore_eos=True,
    return_token_ids=True,cache_salt=out.name,max_tokens=32,logprobs=5)
seed=json.loads((out/'seed.json').read_text())['response'] if resume else req('seed',dict(options,prompt=prompt))
follow=prompt+seed['choices'][0]['token_ids']+[198]*100
payload=dict(options,prompt=follow)
reference=json.loads((out/'serial-before.json').read_text())['response'] if resume else req('serial-before',payload)
cached=reference['usage']['prompt_tokens_details']['cached_tokens']
# Native mamba_aligned_replay_boundary: exclude the final token, then
# MTP drops one further aligned block. Verify that exact existing contract.
expected_native=max(((len(prompt)-1)//1920-1)*1920,0)
assert cached==expected_native, (cached,expected_native)
t=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
    rows=list(pool.map(lambda i:req('c10-'+str(i),payload),range(10)))
elapsed=time.time()-t
rows.append(req('serial-after',payload))
checks=[]
for row in rows:
    checks.append({'tokens_equal':row['choices'][0]['token_ids']==reference['choices'][0]['token_ids'],
        'scores_equal':row['choices'][0]['logprobs']==reference['choices'][0]['logprobs'],
        'cached':row['usage']['prompt_tokens_details']['cached_tokens']})
after=metrics()
summary={'before':before,'after':after,'cache_boundary':cached,'checks':checks,'c10_seconds':elapsed,
    'probe_output_tps':320/elapsed,'global_store_bytes_delta':after['kv_offload_store_bytes_total']-before['kv_offload_store_bytes_total'],
    'global_load_bytes_delta':after['kv_offload_load_bytes_total']-before['kv_offload_load_bytes_total'],
    'scope':'Ordinary background workload remains running. Global counters include its activity.'}
summary['passed']=all(c['tokens_equal'] and c['scores_equal'] and c['cached']==cached for c in checks) and after['kv_completion_resident_blocks']==0
(out/'summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
assert summary['passed']
