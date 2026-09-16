import json,urllib.request,time,concurrent.futures,statistics
from pathlib import Path
p=Path(__file__).resolve().parent;out=p/'reuse-benchmark-v2';out.mkdir(exist_ok=False);key=Path('/home/jon/.config/qwen38/api-key').read_text().strip();base='http://127.0.0.1:30001'
source=json.loads((p/'after-cache-v2/continuation-request.json').read_text());original_salt=source['cache_salt'];partial_salt=original_salt+'-partial';records=[]
def run(name,payload):
 payload=dict(payload,request_id='reusebench-'+name)
 req=urllib.request.Request(base+'/v1/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'});start=time.monotonic()
 with urllib.request.urlopen(req,timeout=300) as r:d=json.load(r)
 row={'name':name,'seconds':time.monotonic()-start,'response':d};(out/(name+'.json')).write_text(json.dumps(row));print(name,round(row['seconds'],3),d['usage']['prompt_tokens_details']['cached_tokens'],flush=True);return row
# Populate a shared prefix shorter than the completed response; it cannot
# provide the repaired completion snapshot at7360 for the following request.
for suffix in ['-a','-b']:
 run('seed-partial'+suffix,dict(source,prompt=source['prompt'][:6401],max_tokens=1,cache_salt=partial_salt+suffix))
summary=[]
for phase,salt in [('partial-a',partial_salt+'-a'),('full-a',original_salt),('full-b',original_salt),('partial-b',partial_salt+'-b')]:
 with urllib.request.urlopen(base+'/metrics',timeout=5) as r:raw=r.read().decode()
 assert all(float(l.rsplit(' ',1)[1])==0 for l in raw.splitlines() if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{')))
 start=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(lambda i:run(phase+'-'+str(i),dict(source,cache_salt=salt)),range(4)))
 wall=time.monotonic()-start;gen=sum(r['response']['usage']['completion_tokens'] for r in rows);computed=sum(r['response']['usage']['prompt_tokens']-r['response']['usage']['prompt_tokens_details']['cached_tokens'] for r in rows)
 summary.append({'phase':phase,'wall_seconds':wall,'generated_tokens':gen,'computed_prompt_tokens':computed,'cached_tokens_per_request':[r['response']['usage']['prompt_tokens_details']['cached_tokens'] for r in rows],'generated_tokens_per_second':gen/wall,'completed_requests_per_second':4/wall});records+=rows
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
