from pathlib import Path
import subprocess,json,hashlib,shlex,time,urllib.request
p=Path(__file__).resolve().parent;root=p.parents[1]
# Verify every required runtime source before interrupting service.
for rank in [0,1]:
 cfg=json.loads((p/f'candidate-content-r{rank}.json').read_text())
 for key,path in cfg['runtime_overrides'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==cfg['runtime_override_sha256'][key]
for f in ['content_store.py','rank_local_disk.content.py','candidate-content-r1.json']:
 subprocess.run(['ssh','192.168.100.11','mkdir','-p',str(p)],check=True)
 subprocess.run(['scp',str(p/f),'192.168.100.11:'+str(p/f)],check=True)
m='Preparing transport-only disk-content dedup candidate now: exact per-group state fingerprints share backing files, retaining all logical checkpoints. CPU tests and real-file transport tests (simulated unchanged DMA) pass; no model/kernel/scheduler changes. Will pause current run.py processes and repeat controller by verified PID identities for controlled restart/validation, then resume unchanged. Cycle0 already spans earlier transitions. Please do not independently resume until completion notification. Monitor deadline09:00UTC confirmed; leaving your loop deadline10:00UTC unchanged.'
subprocess.run(['ssh','jon@192.168.0.167',shlex.join(['codex','queue','--thread','01a09b9f-c6c7-7bb0-8c25-b22748f43388','--message',m])],check=True,timeout=30)
remote='''import json,os,signal,time
from pathlib import Path
root=Path('/home/jon/artifact-agent-orchestration/.artifact-agent/baseline-comparison');status=json.loads((root/'20260914-temperature-until-1000/status.json').read_text());record=Path(status['current']);controller=int(status['pid']);selected={}
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  args=(proc/'cmdline').read_bytes().split(b'\\0'); fields=(proc/'stat').read_text().rsplit(') ',1)[1].split()
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if int(proc.name)==controller or (str(record/'run.py').encode() in args and str(record).encode() in args):
  selected[proc.name]={'starttime':fields[19],'state':fields[0],'controller':int(proc.name)==controller,'supervisor':b'--arm' not in args}
for pid,info in sorted(selected.items(),key=lambda x:(not x[1]['controller'],not x[1]['supervisor'])):
 fields=(Path('/proc')/pid/'stat').read_text().rsplit(') ',1)[1].split();assert fields[19]==info['starttime'];os.kill(int(pid),signal.SIGSTOP)
out={'time':time.time(),'record':str(record),'processes':selected,'reason':'controlled content-dedup server validation'}
(root/'content-dedup-validation-pause.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
'''
r=subprocess.run(['ssh','jon@192.168.0.167','python3 -'],input=remote,text=True,capture_output=True,check=True,timeout=30);(p/'content-validation-pause.json').write_text(r.stdout);print(r.stdout,flush=True)
with urllib.request.urlopen('http://127.0.0.1:30001/metrics',timeout=5) as r:(p/'baseline-final-metrics.txt').write_bytes(r.read())
(p/'rollback-r0.json').write_bytes((root/'deploy_config.json').read_bytes())
(root/'deploy_config.json').write_bytes((p/'candidate-content-r0.json').read_bytes())
subprocess.run(['scp',str(p/'candidate-content-r1.json'),'192.168.100.11:'+str(root/'deploy_config.json')],check=True)
(p/'content-deploy-transition.json').write_text(json.dumps({'time':time.time(),'config':'C-content-dedup','pause':'content-validation-pause.json'},indent=2))
subprocess.run(['sudo','-n','systemctl','restart','--no-block','qwen38-next-qwen-fp8.service'],check=True)
print('Content dedup candidate restart requested; numerical/completion kernels unchanged.',flush=True)
