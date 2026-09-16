"""Exact completion reuse and C4 equality under ordinary background traffic."""
import json,time,urllib.request,concurrent.futures,sys
from pathlib import Path
p=Path(__file__).resolve().parent;out=p/('probe-'+str(time.time_ns()));out.mkdir()
key=Path('/home/jon/.config/qwen38/api-key').read_text().strip();base='http://127.0.0.1:30001'
def req(name,payload):
 payload=dict(payload,request_id='eventprobe-'+name)
 r=urllib.request.Request(base+'/v1/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 t=time.time()
 with urllib.request.urlopen(r,timeout=600) as f:d=json.load(f)
 (out/(name+'.json')).write_text(json.dumps({'seconds':time.time()-t,'request':payload,'response':d}));print(name,d.get('usage'),flush=True);return d
rich='rich' in sys.argv
checks=[]
try:
 for index,(length,tail) in enumerate([(128,100),(129,1)] if rich else [(65,100),(96,300),(131,1),(89,100)]):
  salt='eventprobe-'+out.name+'-'+str(index)
  opts={'model':'qwen3.8-flash-next','temperature':0,'seed':917352,'ignore_eos':True,'return_token_ids':True,'cache_salt':salt}
  prompt=(json.loads((p.parent/'aggregate-throughput-20260914/after-J/first.json').read_text())['response']['prompt_token_ids'] if rich else [1012,374,264,1296,13]*200)
  first=req(f'{index}-first',dict(opts,prompt=prompt,max_tokens=length))
  tokens=first['choices'][0]['token_ids'];assert len(tokens)==length
  follow=prompt+tokens+[198]*tail
  payload=dict(opts,prompt=follow,max_tokens=24,logprobs=5)
  reference=req(f'{index}-warm',payload)
  boundary=len(prompt)+length-1
  cached=reference['usage']['prompt_tokens_details']['cached_tokens'];assert cached==boundary,(cached,boundary)
  with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
   rows=list(pool.map(lambda i:req(f'{index}-c4-{i}',payload),range(4)))
  for d in rows:
   assert d['choices'][0]['token_ids']==reference['choices'][0]['token_ids'],'token divergence'
   assert d['choices'][0]['logprobs']==reference['choices'][0]['logprobs'],'score divergence'
   assert d['usage']['prompt_tokens_details']['cached_tokens']==boundary
  checks.append({'boundary':boundary,'offset':boundary%64,'tail':tail,'cached':cached,'comparisons':4,'exact':True})
  (out/'summary.json').write_text(json.dumps({'checks':checks,'finished':False},indent=2))
 (out/'summary.json').write_text(json.dumps({'checks':checks,'passed':True,'finished':True},indent=2))
except BaseException as e:
 (out/'summary.json').write_text(json.dumps({'checks':checks,'passed':False,'error':repr(e),'finished':True},indent=2));raise
print('EVENT GATES PASSED',out,flush=True)
