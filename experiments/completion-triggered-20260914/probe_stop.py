import json,time,urllib.request,concurrent.futures
from pathlib import Path
p=Path(__file__).resolve().parent;out=p/('stop-'+str(time.time_ns()));out.mkdir()
rich=[]
for d in p.glob('probe-*'):
 if d.is_dir() and (d/'0-first.json').exists():
  data=json.loads((d/'0-first.json').read_text())
  if len(data['request']['prompt'])>7000:rich.append(data)
source=rich[-1];prompt=source['request']['prompt'];known=source['response']['choices'][0]['token_ids'];stop=next(t for i,t in enumerate(known) if 12<=i<=90 and t not in known[:i])
key=Path('/home/jon/.config/qwen38/api-key').read_text().strip()
common={'model':'qwen3.8-flash-next','temperature':0,'seed':917352,'return_token_ids':True,'cache_salt':out.name}
def req(name,opts):
 payload=dict(common,**opts,request_id='eventstop-'+name)
 r=urllib.request.Request('http://127.0.0.1:30001/v1/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=300) as f:d=json.load(f)
 (out/(name+'.json')).write_text(json.dumps({'request':payload,'response':d}));return d
first=req('first',{'prompt':prompt,'max_tokens':128,'stop_token_ids':[stop]});c=first['choices'][0];assert c['finish_reason']=='stop',c['finish_reason'];tokens=c['token_ids'];assert len(tokens)>1
payload={'prompt':prompt+tokens+[198]*100,'max_tokens':24,'logprobs':5,'ignore_eos':True}
ref=req('reference',payload);expected=len(prompt)+len(tokens)-1
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(lambda i:req('c4-'+str(i),payload),range(4)))
s={'stop_token':stop,'generated_tokens':len(tokens),'finish_reason':c['finish_reason'],'expected_boundary':expected,'cached':ref['usage']['prompt_tokens_details']['cached_tokens'],'tokens_equal':all(d['choices'][0]['token_ids']==ref['choices'][0]['token_ids'] for d in rows),'scores_equal':all(d['choices'][0]['logprobs']==ref['choices'][0]['logprobs'] for d in rows)}
s['passed']=s['cached']==expected and s['tokens_equal'] and s['scores_equal'];(out/'summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2),flush=True);assert s['passed']
