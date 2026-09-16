import json,time,urllib.request,concurrent.futures,re
from pathlib import Path
p=Path(__file__).parent
original=sorted(p.glob('correctness-*/c1-before.json'))[0]
reference=json.loads(original.read_text())
out=p/('aged-check-'+str(time.time_ns()));out.mkdir()
key=Path('/home/jon/.config/qwen38/api-key').read_text().strip()
def metrics():
 raw=urllib.request.urlopen('http://127.0.0.1:30001/metrics',timeout=10).read().decode()
 return {n:sum(float(x) for x in re.findall(r'^vllm:'+n+r'(?:\{[^\n]*\})? ([\d.eE+-]+)$',raw,re.M)) for n in ['kv_offload_load_bytes_total','kv_offload_store_bytes_total']}
def run(i):
 payload=dict(reference['request']);payload['request_id']=out.name+'-'+str(i)
 before=metrics();start=time.time()
 req=urllib.request.Request('http://127.0.0.1:30001/v1/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as f:response=json.load(f)
 after=metrics();r={'request':payload,'response':response,'start':start,'seconds':time.time()-start,'before':before,'after':after}
 (out/(str(i)+'.json')).write_text(json.dumps(r))
 a=response['choices'][0];b=reference['response']['choices'][0]
 return {'tokens_equal':a['token_ids']==b['token_ids'],'scores_equal':a['logprobs']==b['logprobs'],'cached':response['usage']['prompt_tokens_details']['cached_tokens'],'seconds':r['seconds']}
rows=[run('serial')]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows.extend(pool.map(run,range(4)))
summary={'reference':str(original),'checks':rows,'passed':all(x['tokens_equal'] and x['scores_equal'] for x in rows)}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
assert summary['passed']
