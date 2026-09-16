import json,time,urllib.request,re
from pathlib import Path
p=Path(__file__).resolve().parent;out=p/'disk-oracle-H';out.mkdir(exist_ok=False);base='http://127.0.0.1:30001';key=Path('/home/jon/.config/qwen38/api-key').read_text().strip()
def metrics():
 with urllib.request.urlopen(base+'/metrics') as r:raw=r.read().decode()
 return {name:sum(float(v) for v in re.findall('^vllm:'+name+r'\{[^\n]*\} ([\d.eE+-]+)$',raw,re.M)) for name in ['kv_offload_load_bytes_total','kv_offload_store_bytes_total']}
def call(name,salt,prompt,count):
 payload=dict(model='qwen3.8-flash-next',prompt=prompt,max_tokens=count,temperature=0,seed=917352,ignore_eos=True,return_token_ids=True,logprobs=5,cache_salt=salt,request_id='disk-oracle-'+name)
 req=urllib.request.Request(base+'/v1/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:d=json.load(r)
 (out/(name+'.json')).write_text(json.dumps(d));return d
prompt=[1012,374,264,1296,13]*12
known=json.loads((p/'disk-validation-H/pressure-0-0.json').read_text())['response']['choices'][0]['token_ids'];full=prompt+known+[198];boundary=(len(prompt)+len(known)-1)//64*64
before=metrics();disk=call('disk-restored','H-disk-test-pressure-0-0',full,16);time.sleep(2);after=metrics()
# Independently recreate the same computed boundary under a fresh salt, while
# idle. That finish captures current state to resident memory. Matching prefix
# and full returned continuation scores is required; do not assume cold equivalence.
exact=call('independent-exact','H-independent-oracle',prompt,boundary-len(prompt)+1)
pre=metrics();memory=call('independent-restored','H-independent-oracle',full,16);time.sleep(2);end=metrics()
s={'boundary':boundary,'disk_load_bytes':after['kv_offload_load_bytes_total']-before['kv_offload_load_bytes_total'],'disk_cached':disk['usage']['prompt_tokens_details']['cached_tokens'],'exact_prefix_matches':exact['choices'][0]['token_ids']==known[:boundary-len(prompt)+1],'independent_load_bytes':end['kv_offload_load_bytes_total']-pre['kv_offload_load_bytes_total'],'independent_cached':memory['usage']['prompt_tokens_details']['cached_tokens'],'tokens_equal':disk['choices'][0]['token_ids']==memory['choices'][0]['token_ids'],'scores_equal':disk['choices'][0]['logprobs']==memory['choices'][0]['logprobs']}
(out/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
assert s['disk_load_bytes']>0 and s['independent_load_bytes']==0 and s['disk_cached']==s['independent_cached']==boundary
assert s['exact_prefix_matches'] and s['tokens_equal'] and s['scores_equal']
