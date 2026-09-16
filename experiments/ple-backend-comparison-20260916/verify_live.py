"""Verify mounted source hashes, controls, serving options and all CPU affinities."""

import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
SCRIPT = '''
import hashlib,json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]);rank=int(sys.argv[2])
cfg=json.loads((root/'deploy_config.json').read_text())
container=f'qwen38-kv-paging-r{rank}'
d=json.loads(subprocess.check_output(['docker','inspect',container],text=True))[0]
mounts={m['Destination']:m['Source'] for m in d['Mounts']}
errors=[]
for relative,source in cfg['runtime_overrides'].items():
 if hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest()!=cfg['runtime_override_sha256'][relative]:errors.append('source hash '+relative)
 if mounts.get('/usr/local/lib/python3.12/dist-packages/vllm/'+relative)!=source:errors.append('mount '+relative)
env=dict(x.split('=',1) for x in d['Config']['Env'])
for key,value in {'GB10_PLE_BACKEND_CONTROL':'/opt/gb10/ple-backend-policy.bin','GB10_PLE_IN_PROCESS':'1','GB10_PLE_GPU_HASH':'1','GB10_PLE_OVERLAP':'1','GB10_PLE_SUBMISSION_POLICY':'sqpoll','GB10_PLE_GATHER_THREADS':'16','GB10_PLE_NATIVE_HASH':'1'}.items():
 if env.get(key)!=value:errors.append('environment '+key)
argv=d['Config']['Cmd'];parameters={}
for flag,expected in [('--max-num-seqs','32'),('--max-num-batched-tokens','8192'),('--long-prefill-token-threshold','1024'),('--tensor-parallel-size','2')]:
 value=argv[argv.index(flag)+1] if flag in argv else None;parameters[flag]=value
 if value!=expected:errors.append('serving '+flag)
spec=json.loads(argv[argv.index('--speculative-config')+1])
if spec.get('num_speculative_tokens')!=3:errors.append('MTP setting')
processes=subprocess.check_output(['docker','top',container,'-eo','pid,comm'],text=True).splitlines()[1:]
affinity=[];threads=0
for line in processes:
 pid,name=line.split(maxsplit=1)
 for task in pathlib.Path('/proc',pid,'task').iterdir():
  try:cpus=sorted(os.sched_getaffinity(int(task.name)))
  except ProcessLookupError:continue
  threads+=1
  if cpus!=list(range(20)):affinity.append({'pid':int(pid),'name':name,'tid':int(task.name),'cpus':cpus})
if affinity:errors.append('restricted CPU affinity')
controls={key:int.from_bytes(pathlib.Path(cfg['optimizations'][key]).read_bytes()[:8 if key=='ple_backend_control' else 4],'little') for key in ('ple_backend_control','ple_read_control')}
if controls['ple_read_control']!=0:errors.append('rejected reader enabled')
print(json.dumps({'rank':rank,'container_pid':d['State']['Pid'],'running':d['State']['Running'],'oom_killed':d['State']['OOMKilled'],'restarts':d['RestartCount'],'processes':processes,'threads':threads,'restricted_affinity':affinity,'parameters':parameters,'controls':controls,'errors':errors}))
'''

results = []
for rank in (0, 1):
    command = [str(ROOT / ".venv/bin/python"), "-c", SCRIPT, str(ROOT), str(rank)]
    if rank:
        command = ["ssh", "jon@192.168.100.11", shlex.join(command)]
    result = json.loads(subprocess.check_output(command, text=True, timeout=30))
    results.append(result)
destination = ROOT / "results/ple-backend-comparison-20260916/live-verification.json"
destination.write_text(json.dumps(results, indent=2) + "\n")
assert not any(row["errors"] for row in results), results
assert results[0]["controls"] == results[1]["controls"]
print("PASS both ranks: frozen mounts/source hashes, backend controls, serving settings, unrestricted threads")

