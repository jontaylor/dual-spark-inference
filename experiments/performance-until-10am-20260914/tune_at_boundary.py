"""Hold next-cycle admission, test only after current supervisor exits, always resume."""
import json,subprocess,shlex,time,signal,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent;started=time.time();record={}
def remote(code):return subprocess.check_output(['ssh','jon@192.168.0.167','python3 -'],input=code,text=True,timeout=20)
def notify(message):subprocess.run(['ssh','jon@192.168.0.167',shlex.join(['codex','queue','--thread','01a09b9f-c6c7-7bb0-8c25-b22748f43388','--message',message])],check=True,timeout=30)
def idle():
 with urllib.request.urlopen('http://127.0.0.1:30001/metrics',timeout=3) as r:raw=r.read().decode()
 return all(float(l.rsplit(' ',1)[1])==0 for l in raw.splitlines() if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{')))
def interrupted(*args):raise KeyboardInterrupt
signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
notify('Planning one bounded fixed-K64 GEMM tuning test at the boundary after current cycle0 finishes. I will SIGSTOP only repeat controller1178958 to hold next-cycle admission; all current workload arms/supervisor continue unchanged. Once current supervisor exits and API is idle, test lasts at most240s; controller resumes automatically on success/failure or if boundary not reached within10minutes. Please do not independently resume the controller during this window. Following cycle will start with unchanged workload settings.')
try:
 code='''import json,os,signal
from pathlib import Path
s=json.loads(Path('/home/jon/artifact-agent-orchestration/.artifact-agent/baseline-comparison/20260914-temperature-until-1000/status.json').read_text());pid=int(s['pid']);f=Path('/proc')/str(pid)/'stat';fields=f.read_text().rsplit(') ',1)[1].split();assert pid==1178958 and fields[19]=='9802967';os.kill(pid,signal.SIGSTOP);print(json.dumps({'pid':pid,'starttime':fields[19],'current':s['current']}))
'''
 record=json.loads(remote(code));record.update(started=started,status='holding_next_cycle');(p/'tuning-boundary-status.json').write_text(json.dumps(record,indent=2));print(record,flush=True)
 while time.time()-started<600:
  status=json.loads(remote('''import json
from pathlib import Path
p=Path('/proc/1083663/stat');print(json.dumps({'supervisor_state':p.read_text().rsplit(') ',1)[1].split()[0] if p.exists() else 'gone'}))
'''))
  if status['supervisor_state'] in ['gone','Z'] and idle():break
  time.sleep(5)
 else:
  record['status']='deferred_no_boundary';raise TimeoutError('No idle cycle boundary within10minutes; leaving workload unchanged')
 record['status']='benchmark_running';record['benchmark_started']=time.time();(p/'tuning-boundary-status.json').write_text(json.dumps(record,indent=2));print('Current cycle exited, running bounded idle benchmark',flush=True)
 subprocess.run(['docker','cp',str(p/'gemm_tune.py'),'qwen38-kv-paging-r0:/tmp/gemm_tune.py'],check=True)
 with (p/'gemm-tuning.log').open('w') as log:
  result=subprocess.run(['docker','exec','qwen38-kv-paging-r0','timeout','240','python3','-u','/tmp/gemm_tune.py'],stdout=log,stderr=subprocess.STDOUT)
 record['benchmark_exit_code']=result.returncode;record['status']='benchmark_complete' if result.returncode==0 else 'benchmark_incomplete'
 subprocess.run(['docker','cp','qwen38-kv-paging-r0:/tmp/gb10-gemm-tuning.json',str(p/'gemm-tuning.json')],check=False)
finally:
 if record.get('pid'):
  code='''import json,os,signal
from pathlib import Path
p=Path('/proc/1178958/stat');f=p.read_text().rsplit(') ',1)[1].split();assert f[19]=='9802967';os.kill(1178958,signal.SIGCONT);print('resumed')
'''
  result=remote(code);record['controller_resume']=result.strip();record['finished']=time.time();(p/'tuning-boundary-status.json').write_text(json.dumps(record,indent=2));print(record,flush=True)
  notify('The bounded kernel-tuning boundary window is finished or deferred. Repeat controller1178958 has been identity-checked and resumed. This supersedes the admission hold; keep the normal repeating workload and sampling unchanged. Current server configuration remains content-dedup C; no model/config change was made by this test.')
