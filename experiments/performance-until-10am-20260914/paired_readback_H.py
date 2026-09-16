"""Live ABBA comparison; fixed workload untouched, bounded probe per phase."""
import datetime,hashlib,json,os,re,shlex,signal,subprocess,time,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1];out=p/'readback-paired-H';out.mkdir(exist_ok=False)
expected='ee5e88d3f660cbe261dd8ec4b6ee78bcc17abb6a50d4aabfd59d92347fc7d35d';key=Path('/home/jon/.config/qwen38/api-key').read_text().strip();base='http://127.0.0.1:30001';salt='H-readback-paired-'+str(time.time_ns());record={'started':time.time(),'phases':[],'scope':'Live ABBAABBA120s windows; background frozen eight-arm workload continues. One cached correctness probe BEFORE each measured phase; no client pause or restart. CPU file checksum remains unconditional.'};deadline=datetime.datetime(2026,9,14,5,16,12,tzinfo=datetime.timezone.utc).timestamp();changed=False
(out/'process.json').write_text(json.dumps({'pid':os.getpid(),'starttime':Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ',1)[1].split()[19]},indent=2))
def save():
 tmp=out/'summary.tmp';tmp.write_text(json.dumps(record,indent=2));tmp.replace(out/'summary.json')
def guard():assert hashlib.sha256((root/'deploy_config.json').read_bytes()).hexdigest()==expected,'Serving config changed; stop paired comparison'
def control(enabled):
 global changed
 changed=True;stamp=time.time();data=json.dumps({'gpu_readback':enabled})
 for rank in (0,1):
  code="from pathlib import Path;import sys,json;d=sys.stdin.read();assert type(json.loads(d)['gpu_readback']) is bool;p=Path('/tmp/vllm-kv-readback-control.json');t=p.with_suffix('.tmp');t.write_text(d);t.replace(p);assert json.loads(p.read_text())['gpu_readback']=="+repr(enabled)
  cmd=['docker','exec','-i',f'qwen38-kv-paging-r{rank}','python3','-c',code]
  if rank:cmd=['ssh','jon@192.168.100.11',shlex.join(cmd)]
  subprocess.run(cmd,input=data,text=True,check=True,timeout=20)
 event={'started':stamp,'confirmed':time.time(),'gpu_readback':enabled}
 with (out/'controls.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
 (p/'readback-control-state.json').write_text(json.dumps(event,indent=2))
def metrics():
 with urllib.request.urlopen(base+'/metrics',timeout=5) as r:raw=r.read().decode()
 values={l.rsplit(' ',1)[0]:float(l.rsplit(' ',1)[1]) for l in raw.splitlines() if l.startswith('vllm:') and '_bucket{' not in l and '_created{' not in l and 'config_info{' not in l}
 return {'time':time.time(),'metrics':values}
def request(name,prompt,count):
 payload=dict(model='qwen3.8-flash-next',prompt=prompt,max_tokens=count,temperature=0,seed=917352,ignore_eos=True,return_token_ids=True,logprobs=5,cache_salt=salt,request_id='readback-'+name)
 req=urllib.request.Request(base+'/v1/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=180) as r:d=json.load(r)
 (out/(name+'.json')).write_text(json.dumps(d));return d

def interrupt(*args):raise KeyboardInterrupt
signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
try:
 save();print('Paired readback scheduled for05:16:12UTC; no controls changed yet',flush=True)
 while time.time()<deadline:guard();time.sleep(min(5,deadline-time.time()))
 guard();control(True);prompt=[1012,374,264,1296,13]*12;seed=request('seed',prompt,69);follow=prompt+seed['choices'][0]['token_ids']+[198];reference=request('reference',follow,16);assert reference['usage']['prompt_tokens_details']['cached_tokens']==128
 for index,enabled in enumerate((True,False,False,True,True,False,False,True)):
  guard();control(enabled);probe=request(f'probe-{index}',follow,16)
  assert probe['usage']['prompt_tokens_details']['cached_tokens']==128,'Probe checkpoint evicted; coverage incomplete'
  assert all(probe['choices'][0][k]==reference['choices'][0][k] for k in ('token_ids','logprobs')),'Cached tokens/scores changed'
  before=metrics();record.update(status='measuring',phase=index,gpu_readback=enabled);save();print('Phase',index,'gpu_readback',enabled,flush=True)
  target=time.monotonic()+120
  while time.monotonic()<target:guard();time.sleep(min(5,target-time.monotonic()))
  after=metrics();d={k:v-before['metrics'].get(k,0) for k,v in after['metrics'].items()}
  def total(n):return sum(v for k,v in d.items() if k.startswith('vllm:'+n+'{'))
  secs=after['time']-before['time'];row={'index':index,'gpu_readback':enabled,'start':before['time'],'end':after['time'],'seconds':secs,'generated_tokens':total('generation_tokens_total'),'aggregate_tps':total('generation_tokens_total')/secs,'prompt_tokens':total('prompt_tokens_total'),'cached_tokens':total('prompt_tokens_cached_total'),'completed':total('request_success_total'),'preemptions':total('num_preemptions_total'),'load_bytes':total('kv_offload_load_bytes_total'),'store_bytes':total('kv_offload_store_bytes_total'),'load_time':total('kv_offload_load_time_total'),'store_time':total('kv_offload_store_time_total'),'probe_exact':True}
  (out/f'metrics-{index}.json').write_text(json.dumps({'before':before,'after':after}));assert min(row['generated_tokens'],row['prompt_tokens'],row['load_bytes'])>=0,'Counter reset';record['phases'].append(row);save();print(json.dumps(row),flush=True)
 record['status']='complete'
except BaseException as e:record.update(status='failed',error=repr(e));raise
finally:
 if changed:
  control(True);record['restored_full_verification']=True
 record['finished']=time.time();save()
