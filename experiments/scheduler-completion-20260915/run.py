import concurrent.futures,datetime,fcntl,gzip,hashlib,json,os,pathlib,shlex,subprocess,time,urllib.request,traceback
P=pathlib.Path(__file__).resolve().parent
ROOT=P.parents[1];SERVICE='qwen38-next-qwen-fp8.service'
REMOTE='jon@192.168.0.167';CONTROL='/home/jon/artifact-agent-orchestration/.artifact-agent/restart-reset-repeat/status.json'
MATRIX=json.loads((P/'matrix.json').read_text())
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
def completion_snapshot(record):
 code = """import pathlib,json,sys,time
p=pathlib.Path(sys.argv[1]);m=json.loads((p/'campaign.json').read_text());rows=[]
assert len(m['conditions'])==16
for c in m['conditions']:
 f=p/c['id']/'run.json'
 try:d=json.loads(f.read_text())
 except FileNotFoundError:d={}
 rows.append({'id':c['id'],'status':d.get('status','not_started'),'elapsed_seconds':d.get('elapsed_seconds'),'error':d.get('error'),'usage':d.get('usage'),'checks':d.get('checks'),'mtime':f.stat().st_mtime if f.exists() else None})
print(json.dumps({'remote_time':time.time(),'campaign':str(p),'manifest_status':m['status'],'rows':rows}))
"""
 return json.loads(remote(code,record))
def completed(snapshot):return sum(r['status']=='complete' for r in snapshot['rows'])
def archive_previous(record):
 location=P/(pathlib.Path(record).name+'.archive-location.json')
 if location.exists():
  for item in json.loads(location.read_text()):
   assert call(['sha256sum',item['rank1_path']],1,timeout=600).split()[0]==item['sha256']
  return
 # Existing utility verifies terminal lifecycle, process absence and every file hash before removal.
 subprocess.run(['python3',str(ROOT/'experiments/scheduler-campaign-20260915/archive_inactive.py'),pathlib.Path(record).name],check=True,timeout=600)
 source=ROOT/'experiments/scheduler-campaign-20260915'
 destination=str(P/'full-archives')
 call(['mkdir','-p',destination],1)
 moved=[]
 for suffix in ('.full.tar.gz','.archive-manifest.json'):
  f=source/(pathlib.Path(record).name+suffix)
  digest=hashlib.file_digest(f.open('rb'),'sha256').hexdigest()
  subprocess.run(['scp','-q',str(f),'jon@192.168.100.11:'+destination+'/'],check=True,timeout=600)
  target=destination+'/'+f.name
  actual=call(['sha256sum',target],1,timeout=600).split()[0]
  assert actual==digest
  moved.append({'source':str(f),'rank1_path':target,'sha256':digest});f.unlink()
 save(P/(pathlib.Path(record).name+'.archive-location.json'),moved)
def main():
 lock=(P/'lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 resume=int(__import__('sys').argv[1]) if len(__import__('sys').argv)>1 else 1
 if resume>1:
  base=[json.loads((P/f'baseline-r{r}.json').read_text()) for r in (0,1)]
  for i in range(1,resume):assert any(P.glob(f'{i:02d}-*/complete.json')) or any(P.glob(f'{i:02d}-*/skipped.json')) or any(P.glob(f'{i:02d}-*/failed.json')),'Earlier arm incomplete'
  expected=[json.loads(call(['cat',str(ROOT/'deploy_config.json')],r)) for r in (0,1)]
 else:
  assert not (P/'baseline-r0.json').exists(),'Campaign already started; inspect status before resuming'
  base=[json.loads(call(['cat',str(ROOT/'deploy_config.json')],r)) for r in (0,1)]
  for r,c in enumerate(base):save(P/f'baseline-r{r}.json',c)
  expected=base
 save(P/'matrix.json',MATRIX)
 for n,(seq,budget,threshold) in enumerate(MATRIX,1):
  if n<resume:continue
  if any(P.glob(f'{n:02d}-*/skipped.json')):continue
  policy=json.loads((P/'final-run-policy.json').read_text()) if (P/'final-run-policy.json').exists() else {}
  if n>policy.get('final_arm',len(MATRIX)):break
  target=policy.get('completion_targets',{}).get(str(n),8)
  out=P/f'{n:02d}-s{seq}-b{budget}-t{threshold}';out.mkdir();c=controller();old=c['current']
  free=int(remote('import shutil;print(shutil.disk_usage("/home/jon").free)'));assert free>=2*2**30,'Load host disk below controller launch gate'
  cfgs=json.loads(json.dumps(base))
  for r,cfg in enumerate(cfgs):
   cfg.update(max_num_seqs=seq,max_num_batched_tokens=budget,long_prefill_token_threshold=threshold);save(out/f'config-r{r}.json',cfg)
  logevent(arm=n,phase='stopping',settings=[seq,budget,threshold])
  subprocess.run(['sudo','-n','systemctl','stop',SERVICE],check=True,timeout=180)
  # Guarantee two failed health checks even if shutdown was unusually fast.
  deadline=time.time()+180
  while time.time()<deadline:
   c=controller()
   if not c.get('last_health_ok',True) and c.get('bad_checks',0)>=2 and any(x['record']==old and x['outcome']!='running' for x in c['cycles']):break
   time.sleep(3)
  else:raise RuntimeError('Load controller did not observe outage')
  write_configs(cfgs,expected);expected=cfgs
  logevent(arm=n,phase='archiving_previous',settings=[seq,budget,threshold])
  archive_previous(old)
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
   last_completion_poll=0;snapshot=None;reached=False
   with (out/'completion-history.jsonl').open('a') as history:
    pass
   while not reached:
    t=time.time()
    try:m=metrics()
    except Exception as e:raise RuntimeError('API metrics unavailable during measured arm') from e
    f.write(json.dumps({'time':t,'metrics':m})+'\n');f.flush()
    if start is None:
     c=controller()
     if c['current']!=old and c['status']=='running' and count(m,'num_requests_running')+count(m,'num_requests_waiting')>0:
      start=t;record=c['current'];save(out/'window.json',{'start':start,'completion_target':target,'campaign':record,'controller_cycle_start':c['cycles'][-1]['started_utc'],'clock':'first observed active request in fresh campaign; polling resolution ~2 seconds'})
      logevent(arm=n,phase='measuring',start=start,completion_target=target,campaign=record)
     elif t>deadline:raise RuntimeError('Fresh campaign did not begin')
    if start is not None and t-last_completion_poll>=5:
     c=controller();assert c['current']==record,'Campaign changed before requested completions'
     snapshot=completion_snapshot(record);snapshot['observed_time']=time.time()
     with (out/'completion-history.jsonl').open('a') as history:history.write(json.dumps(snapshot)+'\n')
     last_completion_poll=t
     done=completed(snapshot)
     logevent(arm=n,phase='measuring',settings=[seq,budget,threshold],start=start,campaign=record,completed=done,target=target)
     reached=done>=target
     if reached:
      save(out/'completion-cutoff.json',snapshot)
      save(out/'cutoff.json',{'time':time.time(),'metrics':metrics()})
      break
     failed=sum(r['status'] in ('incomplete','interrupted') for r in snapshot['rows'])
     assert failed<=16-target,'Requested successful completion target is unattainable: failed/interrupted tasks'
     assert c['status']=='running','Workload controller is no longer running'
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
  c=controller();save(out/'controller-cutoff.json',c)
  for r in (0,1):
   s=call(['docker','logs',f'qwen38-kv-paging-r{r}'],r,stderr=subprocess.STDOUT)
   with gzip.open(out/f'server-r{r}.log.gz','wt') as f:f.write(s)
  capture_campaign(out,c);save(out/'complete.json',{'time':time.time(),'campaign_at_start':record,'campaign_at_end':c['current'],'same_campaign':record==c['current'],'completed':completed(snapshot),'completion_target':target});logevent(arm=n,phase='arm_complete')
 logevent(phase='all_requested_arms_complete',note='Final configuration remains running; analysis pending')
if __name__=='__main__':
 try:main()
 except BaseException as e:
  logevent(phase='failed',error=repr(e),traceback=traceback.format_exc());raise
