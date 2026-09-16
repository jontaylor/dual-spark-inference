import concurrent.futures,json,re,time,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent;out=p/'disk-validation';out.mkdir(exist_ok=False)
key=Path('/home/jon/.config/qwen38/api-key').read_text().strip();base='http://127.0.0.1:30001';salt='content-dedup-disk-test'
def metrics():
 with urllib.request.urlopen(base+'/metrics',timeout=5) as r:raw=r.read().decode()
 names=['kv_offload_store_bytes_total','kv_offload_load_bytes_total','kv_completion_resident_blocks','kv_completion_spilling_blocks','num_preemptions_total','num_requests_running','num_requests_waiting']
 return {n:sum(float(x) for x in re.findall(r'^vllm:'+n+r'\{[^\n]*\} ([\d.eE+-]+)$',raw,re.M)) for n in names}
def req(name,prompt,tokens=128,scores=False):
 payload={'model':'qwen3.8-flash-next','prompt':prompt,'max_tokens':tokens,'temperature':0,'seed':917352,'ignore_eos':True,'return_token_ids':True,'cache_salt':salt+'-'+name,'request_id':'content-pressure-'+name}
 if scores:payload['logprobs']=5
 request=urllib.request.Request(base+'/v1/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'});start=time.monotonic()
 with urllib.request.urlopen(request,timeout=300) as r:d=json.load(r)
 (out/(name+('-warm' if scores else '')+'.json')).write_text(json.dumps({'seconds':time.monotonic()-start,'response':d}));assert d['usage']['completion_tokens']==tokens;return d
before=metrics();assert before['num_requests_running']==before['num_requests_waiting']==0
prompt=[1012,374,264,1296,13]*12
for wave in range(2):
 with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:list(pool.map(lambda i:req(f'seed-{wave}-{i}',prompt,32),range(32)))
source=json.loads((p.parent/'aggregate-throughput-20260914/after-content/first.json').read_text())['response']['prompt_token_ids']
first=req('target',source);followup=source+first['choices'][0]['token_ids']+[198];warm=req('target',followup,16,True)
# Keep initial warm reference separate from later restore using same cache salt.
(out/'target-reference.json').write_text(json.dumps(warm));expected=(len(source)+128-1)//64*64
assert warm['usage']['prompt_tokens_details']['cached_tokens']==expected
waves=[]
for wave in range(6):
 with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:list(pool.map(lambda i:req(f'pressure-{wave}-{i}',prompt),range(32)))
 time.sleep(2);waves.append(metrics());print('Pressure wave',wave+1,waves[-1],flush=True)
pre=metrics();restored=req('target',followup,16,True);time.sleep(2);after=metrics()
summary={'before':before,'waves':waves,'before_restore':pre,'after':after,'expected_cached':expected,'restored_cached':restored['usage']['prompt_tokens_details']['cached_tokens'],'disk_load_bytes':after['kv_offload_load_bytes_total']-pre['kv_offload_load_bytes_total'],'tokens_equal':restored['choices'][0]['token_ids']==warm['choices'][0]['token_ids'],'scores_equal':restored['choices'][0]['logprobs']==warm['choices'][0]['logprobs']}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
assert summary['disk_load_bytes']>0,'Disk restore not exercised'
assert summary['restored_cached']==expected and summary['tokens_equal'] and summary['scores_equal']
assert after['num_preemptions_total']==before['num_preemptions_total']
