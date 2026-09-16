import subprocess,shlex,json
from pathlib import Path
p=Path(__file__).resolve().parent
code='''import json,os,signal,time
from pathlib import Path
root=Path('/home/jon/artifact-agent-orchestration/.artifact-agent/baseline-comparison');record=json.loads((root/'content-dedup-validation-pause.json').read_text());result=[]
for pid,info in sorted(record['processes'].items(),key=lambda x:(x[1]['controller'],x[1]['supervisor'])):
 try:fields=(Path('/proc')/pid/'stat').read_text().rsplit(') ',1)[1].split()
 except FileNotFoundError:result.append({'pid':pid,'status':'exited'});continue
 if fields[19]!=info['starttime']:result.append({'pid':pid,'status':'identity_mismatch'});continue
 os.kill(int(pid),signal.SIGCONT);result.append({'pid':pid,'status':'resumed'})
out={'time':time.time(),'record':record['record'],'results':result};(root/'content-dedup-validation-resume.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
'''
r=subprocess.run(['ssh','jon@192.168.0.167','python3 -'],input=code,text=True,capture_output=True,check=True,timeout=30);(p/'content-validation-resume.json').write_text(r.stdout);print(r.stdout)
m='Controlled server validation finished; recorded workload run.py processes and repeat controller have been resumed after PID identity checks. This supersedes the disk-dedup validation pause request. Keep sampling/retries and repeating cycles unchanged. Server transition evidence is under /home/jon/dual-spark-inference-kv-paging/experiments/performance-until-10am-20260914 on inference head; current cycle spans the transport transition and validation pause, so use later full cycles for matched timing. I will report measured write savings separately.'
subprocess.run(['ssh','jon@192.168.0.167',shlex.join(['codex','queue','--thread','01a09b9f-c6c7-7bb0-8c25-b22748f43388','--message',m])],check=True,timeout=30)
