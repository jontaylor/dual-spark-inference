import json,time,sys,urllib.request,concurrent.futures
from pathlib import Path
p=Path(__file__).resolve().parent;mode=sys.argv[1];out=p/mode;out.mkdir(exist_ok=False)
key=Path('/home/jon/.config/qwen38/api-key').read_text().strip();base='http://127.0.0.1:30001';salt='aligned-check-'+mode

def run(name,endpoint,payload):
 payload=dict(payload,request_id='cachecheck-'+mode+'-'+name)
 req=urllib.request.Request(base+endpoint,data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 start=time.monotonic()
 with urllib.request.urlopen(req,timeout=600) as r:d=json.load(r)
 row={'name':name,'seconds':time.monotonic()-start,'response':d};(out/(name+'.json')).write_text(json.dumps(row,indent=2));print(name,round(row['seconds'],3),d.get('usage'),flush=True);return d
common={'model':'qwen3.8-flash-next','temperature':0,'seed':917352,'max_tokens':128,'ignore_eos':True,'return_token_ids':True,'cache_salt':salt}
first=run('first','/v1/chat/completions',dict(common,messages=[{'role':'user','content':('A detailed record about concurrent model inference and numerical arithmetic.\n'*600)+'Write a long technical explanation with at least fifty numbered points.'}]))
ids=first['prompt_token_ids']+first['choices'][0]['token_ids']+[198]
payload=dict(common,prompt=ids,max_tokens=16,logprobs=5);(out/'continuation-request.json').write_text(json.dumps(payload))
warm=run('warm','/v1/completions',payload)
if mode.startswith('after'):
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:parallel=list(pool.map(lambda i:run('parallel-'+str(i),'/v1/completions',payload),range(4)))
 cold=run('cold','/v1/completions',dict(payload,cache_salt=salt+'-cold'))
 def toks(d):return d['choices'][0]['token_ids']
 def scores(d):return d['choices'][0]['logprobs']
 summary={'warm_cache':warm['usage'].get('prompt_tokens_details'),'expected_boundary':(len(first['prompt_token_ids'])+len(first['choices'][0]['token_ids'])-1)//64*64,'warm_parallel_tokens_equal':all(toks(d)==toks(warm) for d in parallel),'warm_parallel_scores_equal':all(scores(d)==scores(warm) for d in parallel),'cold_warm_tokens_equal':toks(cold)==toks(warm),'cold_warm_scores_equal':scores(cold)==scores(warm)}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
 assert summary['warm_parallel_tokens_equal'] and summary['warm_parallel_scores_equal']
 assert warm['usage']['prompt_tokens_details']['cached_tokens']==summary['expected_boundary']
