import hashlib,json,shlex,subprocess
from pathlib import Path
root=Path('/home/jon/dual-spark-inference-kv-paging');e=root/'experiments/targeted-determinism-20260913'
remote='192.168.100.11'
backup=e/'config-before-r0.json'
if backup.exists():raise RuntimeError('Backup already exists; inspect before rerun')
backup.write_bytes((root/'deploy_config.json').read_bytes())
r=subprocess.run(['ssh',remote,'cat '+str(root/'deploy_config.json')],capture_output=True,check=True)
(e/'config-before-r1.json').write_bytes(r.stdout)
mapping={
'models/qwen4_exp/nvidia/hyperconnection.py':e/'hyperconnection.fixed.py',
'model_executor/determinism/gb10_targeted.py':e/'targeted_kernels.py',
'model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py':e/'gdn.execution.py',
 'third_party/flash_linear_attention/ops/gb10_rmsnorm_fixed.py':e/'rmsnorm_fixed.py',
'v1/core/sched/scheduler.py':e/'scheduler.fixed.py',
'distributed/kv_transfer/kv_connector/v1/gb10_completion.py':e/'completion.fixed.py',
'v1/worker/gpu/model_runner.py':root/'experiments/gdn-trace-20260913/model_runner.py',
'v1/worker/gpu/gb10_row_trace.py':root/'experiments/gdn-trace-20260913/row_trace.py'}
for rank in (0,1):
 cfg=json.loads((e/f'config-before-r{rank}.json').read_text())
 cfg.setdefault('runtime_overrides',{}).update({k:str(v) for k,v in mapping.items()})
 cfg.setdefault('runtime_override_sha256',{}).update({k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in mapping.items()})
 (e/f'candidate-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
subprocess.run(['ssh',remote,'mkdir -p '+str(e)],check=True)
for source in set(mapping.values()):subprocess.run(['scp',str(source),remote+':'+str(source)],check=True)
subprocess.run(['scp',str(e/'candidate-r1.json'),remote+':'+str(root/'deploy_config.json')],check=True)
(root/'deploy_config.json').write_bytes((e/'candidate-r0.json').read_bytes())
message='Deploying scoped batch-invariant HC/GDN GEMMs, fixed GDN RMS rows and canonical GDN execution/grid. Primitive C1/C4 checks passed. Keep load cadence, retries and sampling unchanged; full server validation is next. No request serialization.'
subprocess.run(['ssh','jon@192.168.0.167',shlex.join(['codex','queue','--thread','01a08cc2-ed05-7073-aa5a-7bebae318c0e','--message',message])],check=True,timeout=30)
print('Candidate staged on both ranks; backups in experiment directory')
