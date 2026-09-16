import json,time,urllib.request,re,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;source=sorted(p.glob('disk-pressure-*'))[-1];out=p/('disk-oracle-'+str(time.time_ns()));out.mkdir()
key=Path('/home/jon/.config/qwen38/api-key').read_text().strip();base='http://127.0.0.1:30001'
def metric():
 with urllib.request.urlopen(base+'/metrics',timeout=5) as f:s=f.read().decode()
 return sum(float(x) for x in re.findall(r'^vllm:kv_offload_load_bytes_total\{[^\n]*\} ([\d.eE+-]+)$',s,re.M))
def req(name,prompt,tokens,salt):
 opts={'model':'qwen3.8-flash-next','temperature':0,'seed':917352,'ignore_eos':True,'return_token_ids':True,'logprobs':5,'cache_salt':salt,'prompt':prompt,'max_tokens':tokens,'request_id':'eventoracle-'+name}
 r=urllib.request.Request(base+'/v1/completions',data=json.dumps(opts).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 with urllib.request.urlopen(r,timeout=300) as f:d=json.load(f)
 (out/(name+'.json')).write_text(json.dumps(d));return d
attempts=[]
for wave in reversed(range(4)):
 for i in (31,24,16,8,0):
  record=json.loads((source/f'pressure-{wave}-{i}.json').read_text());prompt=record['request']['prompt'];known=record['response']['choices'][0]['token_ids'];follow=prompt+known+[198]*100;boundary=len(prompt)+len(known)-1
  name=f'{wave}-{i}';pre=metric();started=time.time();d=req(name,follow,24,record['request']['cache_salt']);delta=metric()-pre
  log=subprocess.run(['docker','logs','--since',str(started),'qwen38-kv-paging-r0'],capture_output=True,text=True);raw=log.stdout+log.stderr;(out/(name+'-server.log')).write_text(raw)
  restores=[l for l in raw.splitlines() if 'Completion checkpoint restore request=' in l]
  loads=[json.loads(l.split('GB10_DISK_TRANSFER ',1)[1]) for l in raw.splitlines() if 'GB10_DISK_TRANSFER {' in l and not json.loads(l.split('GB10_DISK_TRANSFER ',1)[1])['store']]
  attempt={'name':name,'cached':d['usage']['prompt_tokens_details']['cached_tokens'],'boundary':boundary,'global_load_bytes':delta,'restore_records':len(restores),'disk_load_jobs':len(loads)};attempts.append(attempt);print(attempt,flush=True)
  if delta<=0 or len(restores)!=1 or 'eventoracle-'+name not in restores[0] or len(loads)!=1 or delta!=loads[0]['bytes']:continue
  independent=req('independent-first',prompt,len(known),out.name)
  assert independent['choices'][0]['token_ids']==known,'Independent prefix differs'
  pre=metric();memory=req('independent-restore',follow,24,out.name);independent_bytes=metric()-pre
  summary={'attempts':attempts,'boundary':boundary,'disk_bytes':delta,'attributed_job':loads[0]['job'],'independent_load_bytes':independent_bytes,'exact_prefix':True,'cached_disk':d['usage']['prompt_tokens_details']['cached_tokens'],'cached_resident':memory['usage']['prompt_tokens_details']['cached_tokens'],'tokens_equal':d['choices'][0]['token_ids']==memory['choices'][0]['token_ids'],'scores_equal':d['choices'][0]['logprobs']==memory['choices'][0]['logprobs']}
  summary['passed']=summary['cached_disk']==summary['cached_resident']==boundary and summary['tokens_equal'] and summary['scores_equal'] and independent_bytes==0
  (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True);assert summary['passed'];raise SystemExit(0)
(out/'summary.json').write_text(json.dumps({'passed':False,'reason':'No unambiguously attributable target disk load','attempts':attempts},indent=2));raise RuntimeError('Disk coverage not established')
