"""Exercise actual generation across J reservation boundaries and cached C1/C4."""
import concurrent.futures,json,re,time,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent;out=p/'growth-validation-J';out.mkdir(exist_ok=False)
base='http://127.0.0.1:30001';key=Path('/home/jon/.config/qwen38/api-key').read_text().strip()
def metric(name):
 with urllib.request.urlopen(base+'/metrics',timeout=5) as r:raw=r.read().decode()
 return sum(float(v) for v in re.findall('^vllm:'+name+r'\{[^\n]*\} ([\d.eE+-]+)$',raw,re.M))
def call(name,prompt,count,salt):
 payload=dict(model='qwen3.8-flash-next',prompt=prompt,max_tokens=count,temperature=0,seed=917352,ignore_eos=True,return_token_ids=True,logprobs=5,cache_salt=salt)
 req=urllib.request.Request(base+'/v1/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=300) as r:d=json.load(r)
 (out/(name+'.json')).write_text(json.dumps(d));assert d['usage']['completion_tokens']==count;return d
assert metric('num_requests_running')==metric('num_requests_waiting')==0
before=metric('num_preemptions_total');rows=[]
for n in [7500,15000]:
 prompt=[1012,374,264,1296,13]*(n//5);salt=f'J-reservation-growth-{n}-{int(time.time())}'
 first=call(f'{n}-growth',prompt,512,salt);tokens=first['choices'][0]['token_ids'];assert n//7680<(n+len(tokens))//7680
 follow=prompt+tokens+[198];ref=call(f'{n}-reference',follow,32,salt)
 with concurrent.futures.ThreadPoolExecutor(4) as pool:parallel=list(pool.map(lambda i:call(f'{n}-c4-{i}',follow,32,salt),range(4)))
 assert all(d['choices'][0]['prompt_token_ids']==ref['choices'][0]['prompt_token_ids']==follow and d['choices'][0]['token_ids']==ref['choices'][0]['token_ids'] and d['choices'][0]['logprobs']==ref['choices'][0]['logprobs'] for d in parallel)
 expected=(n+len(tokens)-1)//64*64;assert ref['usage']['prompt_tokens_details']['cached_tokens']==expected
 rows.append({'prompt_tokens':n,'generated_tokens':512,'crossed_boundary':7680 if n==7500 else 15360,'cached_boundary':expected,'cached_c1_c4_exact':True})
assert metric('num_preemptions_total')==before
s={'passed':True,'rows':rows,'scope':'Generation crosses two reservation growth boundaries; subsequent cached C1/C4 tokens and scores exact. No universal determinism claim.'};(out/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
