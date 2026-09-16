import json,time,urllib.request,concurrent.futures,re
from pathlib import Path
p=Path(__file__).resolve().parent;out=p/('diverse-'+str(time.time_ns()));out.mkdir()
key=Path('/home/jon/.config/qwen38/api-key').read_text().strip();base='http://127.0.0.1:30001'
subjects=['Explain how to implement a robust asynchronous job queue in Python with retries and cancellation.','Explain how a compiler lowers nested closures and handles variable capture.','Develop a detailed plan for measuring database query latency and choosing indexes.','Explain numerical stability in matrix factorizations with worked examples.','Describe a rigorous method for diagnosing packet loss on a Linux network.','Design a versioned document storage API with conflict resolution and audit trails.','Explain the physics of heat transport and derive the diffusion equation.','Compare graph traversal algorithms and derive their complexity with concrete examples.']
def request(name,endpoint,opts):
 r=urllib.request.Request(base+endpoint,data=json.dumps(opts).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=300) as f:d=json.load(f)
 (out/(name+'.json')).write_text(json.dumps({'request':opts,'response':d}));return d
def common(i):return {'model':'qwen3.8-flash-next','temperature':0,'seed':917352,'ignore_eos':True,'return_token_ids':True,'cache_salt':out.name+'-'+str(i)}
def seed(i):return request('seed-'+str(i),'/v1/chat/completions',dict(common(i),messages=[{'role':'user','content':subjects[i]+' Give a thorough technical explanation.'}],max_tokens=128))
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:seeds=list(pool.map(seed,range(8)))
def metrics():
 with urllib.request.urlopen(base+'/metrics',timeout=5) as f:s=f.read().decode()
 return {n:sum(float(x) for x in re.findall(r'^vllm:'+n+r'\{[^\n]*\} ([\d.eE+-]+)$',s,re.M)) for n in ['generation_tokens_total','num_requests_running','num_requests_waiting']}
def continuation(i):
 d=seeds[i];prompt=d['prompt_token_ids']+d['choices'][0]['token_ids']+[198]
 return request('continued-'+str(i),'/v1/completions',dict(common(i),prompt=prompt,max_tokens=512))
before=metrics();t=time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:rows=list(pool.map(continuation,range(8)))
elapsed=time.time()-t;after=metrics()
expected=[len(d['prompt_token_ids'])+len(d['choices'][0]['token_ids'])-1 for d in seeds];cached=[d['usage']['prompt_tokens_details']['cached_tokens'] for d in rows];generated=sum(d['usage']['completion_tokens'] for d in rows)
s={'seconds':elapsed,'probe_tokens':generated,'probe_tps':generated/elapsed,'server_aggregate_tps':(after['generation_tokens_total']-before['generation_tokens_total'])/elapsed,'cached':cached,'expected':expected,'before':before,'after':after,'scope':'C8 diverse warm technical prompts, temperature zero, 512 output tokens/request, ordinary background traffic continuing. Capacity check, not historical matched workload.'};s['cache_gate_passed']=cached==expected
(out/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2),flush=True);assert s['cache_gate_passed']
