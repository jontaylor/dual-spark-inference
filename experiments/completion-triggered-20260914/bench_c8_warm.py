import json,time,urllib.request,concurrent.futures,re
from pathlib import Path
p=Path(__file__).resolve().parent;out=p/('c8warm-'+str(time.time_ns()));out.mkdir()
source=None
for d in sorted(p.glob('probe-*')):
 if d.is_dir() and (d/'0-warm.json').exists():
  r=json.loads((d/'0-warm.json').read_text())
  if len(r['request']['prompt'])>7000:source=r['request']
assert source
prior=sorted(p.glob('c8-*'))[-1]
old=json.loads((prior/'0.json').read_text())['response']
source['prompt']=source['prompt']+old['choices'][0]['token_ids']+[198]
expected=len(source['prompt'])-2
key=Path('/home/jon/.config/qwen38/api-key').read_text().strip();base='http://127.0.0.1:30001'
def metrics():
 with urllib.request.urlopen(base+'/metrics',timeout=5) as f:s=f.read().decode()
 return {n:sum(float(x) for x in re.findall(r'^vllm:'+n+r'\{[^\n]*\} ([\d.eE+-]+)$',s,re.M)) for n in ['generation_tokens_total','num_requests_running','num_requests_waiting']}
def call(i):
 payload=dict(source,max_tokens=512,request_id='eventc8-'+str(i));payload.pop('logprobs',None)
 r=urllib.request.Request(base+'/v1/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 t=time.time()
 with urllib.request.urlopen(r,timeout=300) as f:d=json.load(f)
 row={'start':t,'end':time.time(),'response':d};(out/(str(i)+'.json')).write_text(json.dumps(row));return row
payload=dict(source,max_tokens=1,request_id='eventc8-warm-gate')
r=urllib.request.Request(base+'/v1/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
with urllib.request.urlopen(r,timeout=300) as f:gate=json.load(f)
(out/'warm-gate.json').write_text(json.dumps(gate))
assert gate['usage']['prompt_tokens_details']['cached_tokens']==expected,(gate['usage'],expected)
before=metrics();t=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(call,range(8)))
elapsed=time.time()-t;after=metrics();r={'seconds':elapsed,'submitted_concurrency':8,'prompt_tokens_each':len(source['prompt']),'probe_output_tokens':sum(x['response']['usage']['completion_tokens'] for x in rows),'probe_tps':sum(x['response']['usage']['completion_tokens'] for x in rows)/elapsed,'server_aggregate_tps':(after['generation_tokens_total']-before['generation_tokens_total'])/elapsed,'cache_tokens_each':[x['response']['usage']['prompt_tokens_details']['cached_tokens'] for x in rows],'before':before,'after':after,'scope':'Eight identical warm natural-language prompts at temperature zero, with ordinary background traffic continuing; capacity check, not a matched historical workload comparison.'}
assert all(x['response']['usage']['prompt_tokens_details']['cached_tokens']==expected for x in rows),'Warm cache coverage changed'
(out/'summary.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)
