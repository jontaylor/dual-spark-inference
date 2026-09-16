import subprocess,shlex,json
from pathlib import Path
p=Path(__file__).resolve().parent
remote='''import json,os,signal,time
from pathlib import Path
p=Path('/home/jon/artifact-agent-orchestration/.artifact-agent/baseline-comparison/20260914T004328Z-temperature-recovery-repeat')
record=json.loads((p/'aligned-cache-validation-pause.json').read_text()); results=[]
items=list(record['processes'].items()); items.sort(key=lambda x:x[0]=='1083663')
for pid,expected in items:
 f=Path('/proc')/pid/'stat'
 try: fields=f.read_text().rsplit(') ',1)[1].split()
 except FileNotFoundError: results.append({'pid':pid,'status':'exited'});continue
 if fields[19]!=expected['starttime']:
  results.append({'pid':pid,'status':'identity_mismatch'});continue
 os.kill(int(pid),signal.SIGCONT);results.append({'pid':pid,'status':'resumed'})
out={'time':time.time(),'results':results}
(p/'aligned-cache-validation-resume.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
'''
r=subprocess.run(['ssh','jon@192.168.0.167','python3 -'],input=remote,text=True,capture_output=True,check=True,timeout=30)
(p/'campaign-resume-v2.json').write_text(r.stdout);print(r.stdout)
m='Validation finished; resume normal workload now. All seven recorded campaign processes have been sent SIGCONT after PID identity checks (children first). This supersedes earlier pause requests. V2 aligned completion-cache repair is live with MTP3, unchanged validated scheduler and scoped deterministic kernels. C1/C4 plus seed and mixed C8 tokens/scores pass. Sampling/cadence/retries unchanged. Campaign spans server configurations and validation pauses, so whole-campaign timing is not a single-config benchmark. Measuring post-resume interval cache reuse now.'
subprocess.run(['ssh','jon@192.168.0.167',shlex.join(['codex','queue','--thread','01a09b9f-c6c7-7bb0-8c25-b22748f43388','--message',m])],check=True,timeout=30)
