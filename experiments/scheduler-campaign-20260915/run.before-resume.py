import concurrent.futures,datetime,fcntl,gzip,hashlib,json,os,pathlib,shlex,subprocess,time,urllib.request,traceback
P=pathlib.Path(__file__).resolve().parent
ROOT=P.parents[1];SERVICE='qwen38-next-qwen-fp8.service'
REMOTE='jon@192.168.0.167';CONTROL='/home/jon/artifact-agent-orchestration/.artifact-agent/restart-reset-repeat/status.json'
MATRIX=[(32,16384,2048),(32,16384,512),(32,4096,512),(32,4096,2048),(24,4096,2048),(24,4096,512),(24,16384,512),(24,16384,2048)]
def save(p,d):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def call(args,rank=0,timeout=60,**kw):
 return subprocess.check_output(['ssh','-o','ConnectTimeout=10','jon@192.168.100.11',shlex.join(args)] if rank else args,text=True,timeout=timeout,**kw)
def remote(code,*args):return subprocess.check_output(['ssh','-o','ConnectTimeout=10',REMOTE,shlex.join(['python3','-c',code,*args])],text=True,timeout=40)
def controller():
 return json.loads(remote('import json,pathlib;d=json.loads(pathlib.Path(__import__("sys").argv[1]).read_text());p=pathlib.Path("/proc")/str(d["controller"]["pid"])/"stat";s=p.read_text().split(") ",1)[1].split();assert s[19]==d["controller"]["start_ticks"] and s[0]!="Z";print(json.dumps(d))',CONTROL))
def healthy():
 try:return urllib.request.urlopen('http://127.0.0.1:30001/health',timeout=3).status==200
 except Exception:return False
def metrics():
 raw=urllib.request.urlopen('http://127.0.0.1:30001/metrics',timeout=5).read().decode();m={}
 for l in raw.splitlines():
  if l.startswith('#'):continue
  try:k,v=l.rsplit(' ',1);m[k]=float(v)
  except ValueError:pass
 return m
def count(m,term):return sum(v for k,v in m.items() if k.split('{')[0]=='vllm:'+term)
def runtime(rank):
 d=json.loads(call(['docker','inspect',f'qwen38-kv-paging-r{rank}'],rank))[0]
 return {'state':d['State'],'args':d['Args'],'env':[x for x in d['Config']['Env'] if not any(s in x.split('=')[0].upper() for s in ('KEY','SECRET','TOKEN','PASSWORD'))],'mounts':d['Mounts']}
def telemetry(rank):
 code='import pathlib,json,subprocess;print(json.dumps({"meminfo":pathlib.Path("/proc/meminfo").read_text(),"stat":pathlib.Path("/proc/stat").read_text(),"diskstats":pathlib.Path("/proc/diskstats").read_text(),"gpu":subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu","--format=csv,noheader,nounits"],text=True)}))'
 return json.loads(call(['python3','-c',code],rank,timeout=15))
def capture_campaign(out,c):
 record=c['current'];save(out/'controller.json',c)
 # Preserve result/timing evidence, not source checkouts or tool workspaces.
 code='import pathlib,sys,tarfile;p=pathlib.Path(sys.argv[1]);t=tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz");[(t.add(f,arcname=str(f.relative_to(p)),recursive=False)) for f in p.rglob("*") if f.is_file() and len(f.relative_to(p).parts)<=3 and not any(x in f.relative_to(p).parts for x in ("source","current-source","initial-project","final-project",".git")) and f.suffix in (".json",".jsonl",".log",".txt")];t.close()'
 with (out/'campaign-evidence.tar.gz').open('wb') as f:
  subprocess.run(['ssh',REMOTE,shlex.join(['python3','-c',code,record])],stdout=f,check=True,timeout=120)
def write_configs(configs,expected):
 for r,c in enumerate(configs):
  code='import sys,pathlib,json;p=pathlib.Path(sys.argv[1]);d=json.load(sys.stdin);assert json.loads(p.read_text())==d["expected"],"config drift";q=p.with_suffix(".campaign-tmp");q.write_text(json.dumps(d["new"],indent=2)+chr(10));q.replace(p)'
  call(['python3','-c',code,str(ROOT/'deploy_config.json')],r,input=json.dumps({'expected':expected[r],'new':c}))
def logevent(**kw):
 save(P/'status.json',{'pid':os.getpid(),'time':time.time(),**kw});print(json.dumps(kw),flush=True)
def main():
 lock=(P/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (P/'baseline-r0.json').exists(),'Campaign already started; inspect status before resuming'
 base=[json.loads(call(['cat',str(ROOT/'deploy_config.json')],r)) for r in (0,1)]
 for r,c in enumerate(base):save(P/f'baseline-r{r}.json',c)
 save(P/'matrix.json',MATRIX);expected=base
 for n,(seq,budget,threshold) in enumerate(MATRIX,1):
  out=P/f'{n:02d}-s{seq}-b{budget}-t{threshold}';out.mkdir();c=controller();old=c['current']
  free=int(remote('import shutil;print(shutil.disk_usage("/home/jon").free)'));assert free>=2*2**30,'Load host disk below controller launch gate'
  cfgs=json.loads(json.dumps(base))
  for r,cfg in enumerate(cfgs):
   cfg.update(max_num_seqs=seq,max_num_batched_tokens=budget,long_prefill_token_threshold=threshold);save(out/f'config-r{r}.json',cfg)
  logevent(arm=n,phase='stopping',settings=[seq,budget,threshold])
  subprocess.run(['sudo','-n','systemctl','stop',SERVICE],check=True,timeout=180)
  # Guarantee two failed health checks even if shutdown was unusually fast.
  deadline=time.time()+90
  while time.time()<deadline:
   c=controller()
   if not c.get('last_health_ok',True) and c.get('bad_checks',0)>=2:break
   time.sleep(3)
  else:raise RuntimeError('Load controller did not observe outage')
  write_configs(cfgs,expected);expected=cfgs
  subprocess.run(['sudo','-n','systemctl','start','--no-block',SERVICE],check=True,timeout=30)
  logevent(arm=n,phase='loading',settings=[seq,budget,threshold])
  deadline=time.time()+1200
  while not healthy():
   if time.time()>deadline:raise RuntimeError('Readiness timeout; inspect live service before retry')
   active=subprocess.run(['systemctl','is-active',SERVICE],capture_output=True,text=True).stdout.strip()
   if active not in ('active','activating'):raise RuntimeError('Service stopped during loading')
   time.sleep(5)
  for r in (0,1):
   d=runtime(r);save(out/f'runtime-r{r}.json',d);a=d['args']
   for flag,v in [('--max-num-seqs',seq),('--max-num-batched-tokens',budget),('--long-prefill-token-threshold',threshold)]:assert a[a.index(flag)+1]==str(v)
  logevent(arm=n,phase='waiting_for_fresh_campaign')
  deadline=time.time()+240;start=None;record=None
  with gzip.open(out/'metrics.jsonl.gz','wt') as f, gzip.open(out/'hardware.jsonl.gz','wt') as hw,concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
   last_hardware=0;profiles=set();profile_processes=[]
   while start is None or time.time()<start+600:
    t=time.time()
    try:m=metrics()
    except Exception as e:raise RuntimeError('API metrics unavailable during measured arm') from e
    f.write(json.dumps({'time':t,'metrics':m})+'\n');f.flush()
    if start is None:
     c=controller()
     if c['current']!=old and c['status']=='running' and count(m,'num_requests_running')+count(m,'num_requests_waiting')>0:
      start=t;record=c['current'];save(out/'window.json',{'start':start,'end_planned':start+600,'campaign':record,'controller_cycle_start':c['cycles'][-1]['started_utc'],'clock':'first observed active request in fresh campaign; polling resolution ~2 seconds'})
      logevent(arm=n,phase='measuring',start=start,end=start+600,campaign=record)
     elif t>deadline:raise RuntimeError('Fresh campaign did not begin')
    if start is not None:
     for offset in (120,420):
      if t-start>=offset and offset not in profiles:
       profiles.add(offset)
       top=call(['docker','top','qwen38-kv-paging-r0','-eo','pid,args'])
       for line in top.splitlines():
        if 'VLLM::EngineCore' in line or 'VLLM::Worker_TP0' in line:
         pid=line.split()[0];label='engine' if 'EngineCore' in line else 'worker'
         log=(out/f'profile-{offset}-{label}.log').open('w')
         proc=subprocess.Popen(['sudo','-n','/home/jon/.local/bin/py-spy','record','--pid',pid,'--rate','25','--duration','20','--format','speedscope','--idle','--nonblocking','--output',str(out/f'profile-{offset}-{label}.json')],stdout=log,stderr=subprocess.STDOUT)
         profile_processes.append((proc,log))
    if t-last_hardware>=5:
     futures=[ex.submit(telemetry,r) for r in (0,1)]
     for r,fut in enumerate(futures):
      try:hw.write(json.dumps({'time':t,'rank':r,'data':fut.result()})+'\n')
      except Exception as e:hw.write(json.dumps({'time':t,'rank':r,'error':str(e)})+'\n')
     hw.flush();last_hardware=t
    time.sleep(max(0,2-(time.time()-t)))
  for proc,log in profile_processes:
   proc.wait(timeout=30);log.close()
  save(out/'cutoff.json',{'time':time.time(),'metrics':metrics()});c=controller();save(out/'controller-cutoff.json',c)
  for r in (0,1):
   s=call(['docker','logs',f'qwen38-kv-paging-r{r}'],r,stderr=subprocess.STDOUT)
   with gzip.open(out/f'server-r{r}.log.gz','wt') as f:f.write(s)
  capture_campaign(out,c);save(out/'complete.json',{'time':time.time(),'campaign_at_start':record,'campaign_at_end':c['current'],'same_campaign':record==c['current']});logevent(arm=n,phase='arm_complete')
 logevent(phase='eight_arms_complete',note='Final configuration remains running; analysis pending')
if __name__=='__main__':
 try:main()
 except BaseException as e:
  logevent(phase='failed',error=repr(e),traceback=traceback.format_exc());raise
