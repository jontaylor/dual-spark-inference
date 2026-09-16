import concurrent.futures,json,re,time,urllib.request,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;out=p/('disk-pressure-'+str(time.time_ns()));out.mkdir()
key=Path('/home/jon/.config/qwen38/api-key').read_text().strip();base='http://127.0.0.1:30001'
def metrics():
 with urllib.request.urlopen(base+'/metrics',timeout=5) as r:raw=r.read().decode()
 return {n:sum(float(x) for x in re.findall(r'^vllm:'+n+r'\{[^\n]*\} ([\d.eE+-]+)$',raw,re.M)) for n in ['kv_offload_load_bytes_total','kv_offload_store_bytes_total','num_preemptions_total','num_requests_running','num_requests_waiting']}
def req(name,prompt,tokens,salt):
 payload={'model':'qwen3.8-flash-next','prompt':prompt,'max_tokens':tokens,'temperature':0,'seed':917352,'ignore_eos':True,'return_token_ids':True,'logprobs':5,'cache_salt':salt,'request_id':'eventdisk-'+name}
 r=urllib.request.Request(base+'/v1/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 t=time.time()
 with urllib.request.urlopen(r,timeout=600) as f:d=json.load(f)
 (out/(name+'.json')).write_text(json.dumps({'start':t,'end':time.time(),'request':payload,'response':d}));return d
salt=out.name;prompt=[1012,374,264,1296,13]*200
first=req('target-first',prompt,65,salt);follow=prompt+first['choices'][0]['token_ids']+[198]*100
reference=req('target-reference',follow,24,salt);boundary=len(prompt)+64
assert reference['usage']['prompt_tokens_details']['cached_tokens']==boundary
waves=[]
for wave in range(4):
 with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
  rows=list(pool.map(lambda i:req(f'pressure-{wave}-{i}',[1012,374,264,1296,13]*12,32,salt+f'-{wave}-{i}'),range(32)))
 assert all(d['usage']['completion_tokens']==32 for d in rows)
 waves.append(metrics());print('wave',wave,waves[-1],flush=True)
pre=metrics();t=time.time();restored=req('target-restored',follow,24,salt);post=metrics()
summary={'boundary':boundary,'waves':waves,'before_restore':pre,'after_restore':post,'load_bytes_during_restore':post['kv_offload_load_bytes_total']-pre['kv_offload_load_bytes_total'],'cached':restored['usage']['prompt_tokens_details']['cached_tokens'],'tokens_equal':restored['choices'][0]['token_ids']==reference['choices'][0]['token_ids'],'scores_equal':restored['choices'][0]['logprobs']==reference['choices'][0]['logprobs'],'scope':'Background workload continues. Global load-byte deltas require log attribution before claiming target-specific disk reads.'}
log=subprocess.run(['docker','logs','--since',str(int(t)-2),'qwen38-kv-paging-r0'],capture_output=True,text=True);(out/'restore-server.log').write_text(log.stdout+log.stderr)
summary['passed']=summary['cached']==boundary and summary['tokens_equal'] and summary['scores_equal']
(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True);assert summary['passed']
