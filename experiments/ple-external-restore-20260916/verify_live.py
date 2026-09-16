"""Verify the restored external-only runtime on both hosts, without inference."""

import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/ple-external-restore-20260916"
SCRIPT = r'''
import hashlib,json,os,pathlib,re,subprocess,sys
root=pathlib.Path(sys.argv[1]);rank=int(sys.argv[2])
cfg=json.loads((root/'deploy_config.json').read_text())
container=f'qwen38-kv-paging-r{rank}'
d=json.loads(subprocess.check_output(['docker','inspect',container],text=True))[0]
mounts={m['Destination']:m['Source'] for m in d['Mounts']}
env=dict(x.split('=',1) for x in d['Config']['Env'])
errors=[]
for key in ('GB10_PLE_IN_PROCESS','GB10_PLE_GPU_HASH','GB10_PLE_OVERLAP',
            'GB10_PLE_SUBMISSION_POLICY','GB10_PLE_READ_CONTROL',
            'GB10_PLE_BACKEND_CONTROL','GB10_PLE_COMPARISON_LOG_DIR'):
 if key in env:errors.append('unexpected native/comparison environment '+key)
for key,value in {'VLLM_PLE_CPU_OFFLOAD':'1','VLLM_PLE_LOCAL_TP':'1',
                  'GB10_PLE_GATHER_THREADS':'16','GB10_PLE_NATIVE_HASH':'1',
                  'GB10_PLE_MAPPED_TRANSPORT':'1','GB10_PLE_ROW_CACHE_MB':'1024'}.items():
 if env.get(key)!=value:errors.append('external environment '+key)
for dest in mounts:
 if any(x in dest for x in ('libple_batch_reader','libple_hash_gpu',
         'ple-backend-policy','ple-read-policy','/ple-comparison',
         'ple_offload/in_process.py','ple_offload/backend_')):
  errors.append('unexpected native/comparison mount '+dest)
if any('seccomp=' in x for x in (d['HostConfig']['SecurityOpt'] or [])):
 errors.append('custom seccomp remains')
for relative,source in cfg['runtime_overrides'].items():
 if hashlib.sha256(pathlib.Path(source).read_bytes()).hexdigest()!=cfg['runtime_override_sha256'][relative]:
  errors.append('source hash '+relative)
 if mounts.get('/usr/local/lib/python3.12/dist-packages/vllm/'+relative)!=source:
  errors.append('mount '+relative)
argv=d['Config']['Cmd'];parameters={}
for flag,expected in [('--max-num-seqs','32'),('--max-num-batched-tokens','8192'),
                      ('--long-prefill-token-threshold','1024'),('--tensor-parallel-size','2')]:
 value=argv[argv.index(flag)+1] if flag in argv else None;parameters[flag]=value
 if value!=expected:errors.append('serving '+flag)
if json.loads(argv[argv.index('--speculative-config')+1]).get('num_speculative_tokens')!=3:
 errors.append('MTP setting')
processes=subprocess.check_output(['docker','top',container,'-eo','pid,comm'],text=True).splitlines()[1:]
logs=subprocess.run(['docker','logs',container],capture_output=True,text=True,check=True)
worker_namespace_pids=set(re.findall(r'PleOffloadWorker pid=(\d+)',logs.stdout+logs.stderr))
affinity=[];threads=0;external_workers=[];native_libraries=[]
for line in processes:
 pid,name=line.split(maxsplit=1)
 try:tasks=list(pathlib.Path('/proc',pid,'task').iterdir())
 except FileNotFoundError:continue
 try:status=pathlib.Path('/proc',pid,'status').read_text()
 except FileNotFoundError:continue
 namespace_pid=next(line.split()[-1] for line in status.splitlines() if line.startswith('NSpid:'))
 if namespace_pid in worker_namespace_pids:external_workers.append(int(pid))
 for task in tasks:
  try:cpus=sorted(os.sched_getaffinity(int(task.name)))
  except ProcessLookupError:continue
  threads+=1
  if cpus!=list(range(20)):affinity.append({'pid':int(pid),'tid':int(task.name),'cpus':cpus})
 try:
  for entry in subprocess.check_output(['sudo','-n','cat',f'/proc/{pid}/maps'],text=True).splitlines():
   if any(x in entry for x in ('libple_batch_reader.so','libple_hash_gpu.so')):native_libraries.append(entry)
 except subprocess.CalledProcessError:
  if pathlib.Path('/proc',pid).exists():raise
if len(external_workers)!=1:errors.append('expected one original external worker')
if affinity:errors.append('restricted CPU affinity')
if native_libraries:errors.append('native reader/hash still mapped')
code="import hashlib,json,pathlib; p=pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload'); print(json.dumps({n:hashlib.sha256((p/(n+'.py')).read_bytes()).hexdigest() for n in ('connector','worker','protocol')}))"
legacy=json.loads(subprocess.check_output(['docker','exec',container,'python3','-c',code],text=True))
for name,digest in legacy.items():
 expected=hashlib.sha256((root/f'experiments/ple-backend-comparison-20260916/legacy_{name}.py').read_bytes()).hexdigest()
 if digest!=expected:errors.append('original image source differs '+name)
if not d['State']['Running'] or d['State']['OOMKilled'] or d['RestartCount']:
 errors.append('container state')
print(json.dumps({'rank':rank,'container_pid':d['State']['Pid'],
 'running':d['State']['Running'],'oom_killed':d['State']['OOMKilled'],
 'restarts':d['RestartCount'],'processes':processes,'external_workers':external_workers,
 'threads':threads,'restricted_affinity':affinity,'native_libraries':native_libraries,
 'parameters':parameters,'original_source_sha256':legacy,
 'security_options':d['HostConfig']['SecurityOpt'],'errors':errors}))
'''


def main():
    results = []
    for rank in (0, 1):
        command = [str(ROOT / ".venv/bin/python"), "-c", SCRIPT, str(ROOT), str(rank)]
        if rank:
            command = ["ssh", "jon@192.168.100.11", shlex.join(command)]
        results.append(json.loads(subprocess.check_output(command, text=True, timeout=30)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "live-verification.json").write_text(json.dumps(results, indent=2) + "\n")
    assert not any(row["errors"] for row in results), results
    print("PASS both ranks: original external workers, native/comparison assets absent, "
          "source hashes/mounts/settings verified, all threads unrestricted")


if __name__ == "__main__":
    main()
