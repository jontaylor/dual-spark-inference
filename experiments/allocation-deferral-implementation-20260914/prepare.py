"""Freeze candidate identities/configs for review, without restarting serving."""
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parents[1]
FILES={
    'v1/kv_offload/gb10_allocation_deferral.py':'deferral.py',
    'v1/kv_offload/gb10_native_pressure.py':'native_pressure.py',
    'v1/core/block_pool.py':'block_pool.py',
    'v1/core/kv_cache_manager.py':'kv_cache_manager.py',
    'v1/core/sched/scheduler.py':'scheduler.py',
    'v1/kv_offload/rank_local_disk.py':'rank_local_disk.py',
    'distributed/kv_transfer/kv_connector/v1/offloading/common.py':'common.py',
    'distributed/kv_transfer/kv_connector/v1/offloading/worker.py':'worker.py',
    'distributed/kv_transfer/kv_connector/v1/gb10_completion.py':'completion.py',
    'distributed/kv_transfer/kv_connector/v1/gb10_aligned_offloading_connector.py':'connector.py',
    'v1/core/sched/gb10_parking_scheduler.py':'parking_scheduler.py',
    'distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py':'offloading_scheduler.py',
}
def call(rank,args):
    return subprocess.check_output(['ssh','jon@192.168.100.11',shlex.join(args)] if rank else args,text=True,timeout=60)
checks={}
for log in ['ownership-check.log','scheduler-check.log','completion-check.log',
            'hybrid-check.log','connector-check.log','gpu-r0.log','gpu-r1.log']:
    rows=[json.loads(l) for l in (P/log).read_text().splitlines() if l.startswith('{')]
    assert rows and rows[-1]['passed'],log
    checks[log]=rows[-1]
(P/'checks.json').write_text(json.dumps(checks,indent=2))
manifest={dest:{'source':str(P/file),'sha256':hashlib.sha256((P/file).read_bytes()).hexdigest()} for dest,file in FILES.items()}
(P/'manifest.json').write_text(json.dumps(manifest,indent=2))
for rank in (0,1):
    before=json.loads(call(rank,['cat',str(ROOT/'deploy_config.json')]))
    (P/f'before-r{rank}.json').write_text(json.dumps(before,indent=2))
    candidate=json.loads(json.dumps(before))
    for dest,row in manifest.items():
        candidate['runtime_overrides'][dest]=row['source']
        candidate['runtime_override_sha256'][dest]=row['sha256']
    (P/f'candidate-r{rank}.json').write_text(json.dumps(candidate,indent=2))
print('Prepared both rank configurations; live configuration untouched')
