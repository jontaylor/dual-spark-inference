"""Create reviewable replacement/diff artifacts; never install them."""
import difflib
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parents[1]
cfg=json.loads((ROOT/'deploy_config.json').read_text())
files={
 'parking_scheduler.py':'v1/core/sched/gb10_parking_scheduler.py',
 'terminal_worker.py':'v1/kv_offload/gb10_terminal_worker.py',
 'terminal_state.py':'v1/kv_offload/gb10_terminal_state.py',
 'terminal_versions.py':'v1/kv_offload/gb10_terminal_versions.py',
 'completion.py':'distributed/kv_transfer/kv_connector/v1/gb10_completion.py',
 'connector.py':'distributed/kv_transfer/kv_connector/v1/gb10_aligned_offloading_connector.py',
 'gpu_completion_copy.py':'v1/kv_offload/gb10_gpu_completion_copy.py',
 'model_runner.py':'v1/worker/gpu/model_runner.py',
 'launch_rank.py':'launch_rank.py',
}
manifest=[];diff=[]
for file,target in files.items():
 after=P/'candidate'/file
 before=(ROOT/'launch_rank.py' if file=='launch_rank.py' else
         Path(cfg['runtime_overrides'][target]) if target in cfg['runtime_overrides'] else None)
 if file=='parking_scheduler.py' and before is None:
  before=ROOT/'files/paging/gb10_parking_scheduler.py'
 old=before.read_text() if before else ''
 new=after.read_text()
 manifest.append(dict(target=target,candidate=str(after.relative_to(ROOT)),
  before_source=str(before) if before else None,
  before_sha256=hashlib.sha256(old.encode()).hexdigest() if before else None,
  candidate_sha256=hashlib.sha256(new.encode()).hexdigest()))
 label='launch_rank.py' if file=='launch_rank.py' else 'vllm/'+target
 diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
  fromfile='a/'+label if before else '/dev/null',tofile='b/'+label))
(P/'candidate.patch').write_text(''.join(diff))
hotfix='distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py'
# Record all unchanged runtime inputs too: applying this candidate must retain
# the baseline override graph, including the same-pass restore hotfix.
(P/'PATCH_MANIFEST.json').write_text(json.dumps(dict(
 status='candidate only; deployment gates remain in IMPLEMENTATION_STATUS.md',
 replacements=manifest,
 preserve_runtime_overrides={k:v for k,v in cfg['runtime_overrides'].items() if k not in files.values()},
 proposed_config_changes={'async_scheduling':True,'kv_paging.async_terminal_snapshot_bytes':2**31},
 memory_semantics='snapshot reservation is deducted from kv_bytes_per_rank; proposed 40 GiB total becomes 38 GiB KV plus 2 GiB snapshot reservation',
 installation='Replace launch_rank.py only in the repository root. Set each vLLM replacement as a runtime_override and update its SHA256 on both nodes. This manifest does not authorize bypassing validation gates.'),indent=2)+'\n')
print('Wrote candidate.patch and PATCH_MANIFEST.json; serving untouched.')
