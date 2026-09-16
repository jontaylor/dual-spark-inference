"""Exercise actual inherited full-page reuse after displacing local GPU prefixes."""
import concurrent.futures,json,subprocess,time,urllib.request,re
from pathlib import Path
p=Path(__file__).resolve().parent;out=p/'inherited-model-H';out.mkdir(exist_ok=False);key=Path('/home/jon/.config/qwen38/api-key').read_text().strip();base='http://127.0.0.1:30001';start=time.time()
source=json.loads((p.parent/'aggregate-throughput-20260914/after-H/first.json').read_text())['response']['prompt_token_ids'];first=json.loads((p/'disk-validation-H/target.json').read_text())['response'];follow=source+first['choices'][0]['token_ids']+[198];reference=json.loads((p/'disk-validation-H/target-reference.json').read_text())
def call(name,prompt,salt,count,scores=False):
 payload=dict(model='qwen3.8-flash-next',prompt=prompt,temperature=0,seed=917352,max_tokens=count,ignore_eos=True,return_token_ids=True,cache_salt=salt,request_id=name)
 if scores:payload['logprobs']=5
 req=urllib.request.Request(base+'/v1/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:d=json.load(r)
 (out/(name+'.json')).write_text(json.dumps(d));return d
results=[]
for wave in range(3):
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  futures=[pool.submit(call,f'evict-{wave}-{i}',[1012,374,264,1296,13]*3200,f'H-inherit-pressure-{wave}-{i}',1) for i in range(8)]
  for f in futures:f.result()
 name=f'H-inherited-restored-{wave}';restored=call(name,follow,'H-disk-test-target',16,True);time.sleep(1)
 logs=subprocess.run(['docker','logs','--since',str(int(start)),'qwen38-kv-paging-r0'],capture_output=True,text=True,check=True);lines=[l for l in (logs.stdout+logs.stderr).splitlines() if 'Completion snapshot plan' in l and name in l];inherited=max((int(re.search(r'inherited=(\d+)',l).group(1)) for l in lines),default=0)
 row={'wave':wave,'inherited_pages':inherited,'cached':restored['usage']['prompt_tokens_details']['cached_tokens'],'tokens_equal':restored['choices'][0]['token_ids']==reference['choices'][0]['token_ids'],'scores_equal':restored['choices'][0]['logprobs']==reference['choices'][0]['logprobs'],'seconds':time.time()-start};results.append(row);(out/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
 assert row['tokens_equal'] and row['scores_equal'] and row['cached']==7360
 if inherited:break
else:raise AssertionError('Inherited full attention pages not exercised; retain coverage failure')
print('Actual inherited full-page model path exercised with exact continuation.',flush=True)
