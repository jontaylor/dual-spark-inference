import json,hashlib,subprocess,shlex,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1]
def call(rank,args,**kw):return subprocess.check_output(['ssh','jon@192.168.100.11',shlex.join(args)] if rank else args,text=True,timeout=180,**kw)
for rank in [0,1]:
 rows=[json.loads(l) for l in (p/f'gpu-r{rank}.log').read_text().splitlines() if l.startswith('{')];assert rows[-1]['passed']
assert json.loads((p/'pipeline-check.log').read_text())['passed']
for rank in [0,1]:
 before=json.loads(call(rank,['cat',str(root/'deploy_config.json')]))
 (p/f'before-r{rank}.json').write_text(json.dumps(before,indent=2))
 cfg=json.loads(json.dumps(before))
 for dest,file in [('v1/kv_offload/rank_local_disk.py','rank_local_disk.async.py'),('distributed/kv_transfer/kv_connector/v1/gb10_aligned_offloading_connector.py','connector.async.py'),('v1/kv_offload/gb10_staged_evictions.py','pipeline.py')]:
  src=p/file;cfg['runtime_overrides'][dest]=str(src);cfg['runtime_override_sha256'][dest]=hashlib.sha256(src.read_bytes()).hexdigest()
 (p/f'candidate-r{rank}.json').write_text(json.dumps(cfg,indent=2))
 # Verify everything exists with expected content on each rank before restart.
 for dest,src in cfg['runtime_overrides'].items():assert call(rank,['sha256sum',src]).split()[0]==cfg['runtime_override_sha256'][dest],dest
msg='Deploying async eviction staging candidate: preserve GPU source bytes before overwrite, persist/hash in background with8 staging pages217MB/rank; publish disk ACK only on final completion. CPU backpressure/failure tests and real27MB-page overwrite+checked restore on both GPUs passed. One restart, MTP3/model/cache policy unchanged. Preserve workload settings/retries. Will report live validation and post-change stall sample.'
subprocess.run(['ssh','jon@192.168.0.167',shlex.join(['codex','queue','--thread','01a08cc2-ed05-7073-aa5a-7bebae318c0e','--message',msg])],check=True,timeout=30)
subprocess.run(['sudo','-n','systemctl','stop','qwen38-next-qwen-fp8.service'],check=True,timeout=180)
for rank in [0,1]:
 code="from pathlib import Path;import sys,json;p=Path(sys.argv[1]);s=sys.stdin.read();json.loads(s);q=p.with_suffix('.async-tmp');q.write_text(s);q.replace(p)"
 call(rank,['python3','-c',code,str(root/'deploy_config.json')],input=(p/f'candidate-r{rank}.json').read_text())
subprocess.run(['sudo','-n','systemctl','start','--no-block','qwen38-next-qwen-fp8.service'],check=True,timeout=20)
(p/'start.json').write_text(json.dumps({'epoch':time.time(),'changes':['async native eviction source fence','bounded staging8pages','persistence ACK unchanged']},indent=2))
print('Restart requested',flush=True)
