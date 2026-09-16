from pathlib import Path
import shutil,json,hashlib
r=Path('/home/jon/dual-spark-inference-kv-paging'); e=r/'experiments/batch-invariant-20260913'; e.mkdir(exist_ok=True)
shutil.copy2(r/'launch_rank.py',e/'launch_rank.before.py')
shutil.copy2(r/'deploy_config.json',e/'config.diagnostic.json')
c=json.loads((r/'deploy_config.json').read_text())
for k in ('runtime_overrides','runtime_override_sha256'):
 for name in list(c[k]):
  if name.startswith('v1/worker/gpu/'):
   del c[k][name]
(e/'config.clean.json').write_text(json.dumps(c,indent=2)+'\n')
p=e/'batch_invariant.sm12x.py'; name='model_executor/determinism/batch_invariant.py'
c['runtime_overrides'][name]=str(p); c['runtime_override_sha256'][name]=hashlib.sha256(p.read_bytes()).hexdigest(); c['batch_invariant']=True
(r/'deploy_config.json').write_text(json.dumps(c,indent=2)+'\n')
p=r/'launch_rank.py'; s=p.read_text(); marker='for k,v in env.items():cmd.extend'
assert marker in s
s=s.replace(marker,"if cfg.get('batch_invariant', False):\n    env['VLLM_BATCH_INVARIANT'] = '1'\n"+marker)
p.write_text(s)
