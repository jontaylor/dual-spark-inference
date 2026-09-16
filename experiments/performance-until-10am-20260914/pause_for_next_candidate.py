import json,subprocess,shlex,time,urllib.request,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1]
# Check the next candidate and preserve the exact live C configuration first.
for rank in (0,1):
 c=json.loads((p/f'candidate-inherited-content-r{rank}.json').read_text())
 for k,f in c['runtime_overrides'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==c['runtime_override_sha256'][k]
msg='Controlled candidate transition now: seven batch001 arms finished; remaining temperature1.2 has no call/time limit and remains actively generating. We have >1hour C baseline and will pause remaining run.py/supervisor processes by identity, keeping existing controller hold. Last arm will be marked mixed-config; seven finished arms remain C-only. Next steps: stop metadata observers, stop serving workers, bounded no-model fixed-K GEMM benchmark, reload selected candidate and full validation, then resume all recorded processes and controller. Preserve sampler/retries; do not independently resume during validation.'
subprocess.run(['ssh','jon@192.168.0.167',shlex.join(['codex','queue','--thread','01a09b9f-c6c7-7bb0-8c25-b22748f43388','--message',msg])],check=True,timeout=30)
code='''import json,os,signal,time
from pathlib import Path
root=Path('/home/jon/artifact-agent-orchestration/.artifact-agent/baseline-comparison/20260914-temperature-until-1000');s=json.loads((root/'status.json').read_text());record=Path(s['current']);assert record.name=='batch-001';selected={}
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:a=(p/'cmdline').read_bytes().split(b'\\0');f=(p/'stat').read_text().rsplit(') ',1)[1].split()
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if str(record/'run.py').encode() in a and str(record).encode() in a and f[0]!='Z':
  assert f[0]!='T','Already paused process: investigate ownership';selected[p.name]={'starttime':f[19],'state':f[0],'supervisor':b'--arm' not in a}
for pid,info in sorted(selected.items(),key=lambda x:not x[1]['supervisor']):
 f=Path('/proc')/pid/'stat';fields=f.read_text().rsplit(') ',1)[1].split();assert fields[19]==info['starttime'];os.kill(int(pid),signal.SIGSTOP)
out={'time':time.time(),'record':str(record),'processes':selected,'controller_handled_by':'fine-boundary watchdog separate','reason':'Controlled next-candidate validation; unbounded final arm, seven completed C-only'};(root/'next-candidate-pause.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
'''
r=subprocess.check_output(['ssh','jon@192.168.0.167','python3 -'],input=code,text=True,timeout=30);(p/'next-candidate-pause.json').write_text(r);print(r,flush=True)
with urllib.request.urlopen('http://127.0.0.1:30001/metrics',timeout=5) as r:(p/'C-final-metrics.txt').write_bytes(r.read())
(p/'rollback-C-r0.json').write_bytes((root/'deploy_config.json').read_bytes());worker=subprocess.check_output(['ssh','jon@192.168.100.11','cat',str(root/'deploy_config.json')]);(p/'rollback-C-r1.json').write_bytes(worker)
(p/'C-observation-ended.json').write_text(json.dumps({'time':time.time(),'reason':'Candidate transition; final unbounded arm paused','pause':'next-candidate-pause.json'},indent=2))
