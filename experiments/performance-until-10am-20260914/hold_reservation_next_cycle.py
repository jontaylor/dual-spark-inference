"""Hold only next-cycle admission with an identity-checked timed resume."""
import json,subprocess,shlex,time,signal,os
from pathlib import Path
p=Path(__file__).resolve().parent;record={};started=time.time();release=p/'reservation-next-boundary-release';assert not release.exists()
def remote(code):return subprocess.check_output(['ssh','jon@192.168.0.167','python3 -'],input=code,text=True,timeout=20)
def notify(message):subprocess.run(['ssh','jon@192.168.0.167',shlex.join(['codex','queue','--thread','01a09b9f-c6c7-7bb0-8c25-b22748f43388','--message',message])],check=True,timeout=30)
def save():
 t=p/'reservation-next-boundary-status.tmp';t.write_text(json.dumps(record,indent=2));t.replace(p/'reservation-next-boundary-status.json')
def interrupted(*args):raise KeyboardInterrupt
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
notify('Taking over the existing controller-only hold with a new watchdog; deadline extended15minutes to08:30UTC. Batch-003 continues unchanged. Preparing candidate J: same validated I3/MTP5 model kernels and layout, smaller memory reservation steps to reduce premature disk spilling. No active arm will be paused. Controller1178958 is already held and will remain identity-checked; all current workload arms and supervisor continue. Watchdog resumes on release/interruption or by08:30UTC (09:30London). Please do not independently resume during this window. Sampling/retries and controller10:00UTC deadline remain unchanged. I will notify after resume; deployment only at clean idle boundary with enough validation time.')
try:
 code='''import json,os,signal,time
from pathlib import Path
root=Path('/home/jon/artifact-agent-orchestration/.artifact-agent/baseline-comparison/20260914-temperature-until-1000');s=json.loads((root/'status.json').read_text());pid=int(s['pid']);f=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split();assert pid==1178958 and f[19]=='9802967' and f[0]=='T';current=Path(s['current']);assert current.name=='batch-003';launch=json.loads((current/'launch.json').read_text());supervisor=launch['pids']['run.py'];sf=Path(f'/proc/{supervisor}/stat').read_text().rsplit(') ',1)[1].split();r={'pid':pid,'starttime':f[19],'current':str(current),'supervisor':supervisor,'supervisor_starttime':sf[19],'held_at':time.time()};(root/'reservation-next-boundary-pause.json').write_text(json.dumps(r,indent=2));os.kill(pid,signal.SIGSTOP);print(json.dumps(r))
'''
 record={'pid':1178958,'starttime':'9802967'}
 record=json.loads(remote(code));record.update(watchdog_pid=os.getpid(),watchdog_starttime=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ',1)[1].split()[19],started=started,deadline=1789374600.0,status='holding_next_cycle');save();print(json.dumps(record),flush=True)
 while time.time()<1789374600.0 and not release.exists():
  code=f'''import json
from pathlib import Path
p=Path('/proc/{record['supervisor']}/stat');f=p.read_text().rsplit(') ',1)[1].split() if p.exists() else None;state=f[0] if f and f[19]=={record['supervisor_starttime']!r} else 'gone';m=json.loads(Path({str(Path(record['current'])/'campaign.json')!r}).read_text());print(json.dumps({{'supervisor_state':state,'campaign_status':m['status']}}))
'''
  observed=json.loads(remote(code));record.update(observed=observed,checked_at=time.time());record['status']='boundary_ready' if observed['supervisor_state'] in ('gone','Z') and observed['campaign_status']=='finished' else 'holding_next_cycle';save();print(json.dumps({'status':record['status'],'time':record['checked_at'],'observed':observed}),flush=True)
  for _ in range(30):
   if release.exists():break
   time.sleep(1)
finally:
 if record.get('pid'):
  # Retried remote resume; a transient SSH timeout must not strand admission.
  error=None
  for attempt in range(5):
   try:
    result=remote("import os,signal;from pathlib import Path;f=Path('/proc/1178958/stat').read_text().rsplit(') ',1)[1].split();assert f[19]=='9802967';os.kill(1178958,signal.SIGCONT);print('resumed')")
    record.update(status='resumed',resume=result.strip(),finished=time.time());save();error=None;break
   except Exception as e:error=e;time.sleep(2)
  if error:raise error
  notify('Next-cycle admission hold released. Repeat-controller1178958 identity verified and resumed; continue the original repeating workload, sampling, retries and10:00UTC deadline. This supersedes the earlier hold message.')
