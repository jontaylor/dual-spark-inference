from pathlib import Path
import json,hashlib,shutil
r=Path('/home/jon/dual-spark-inference-kv-paging');e=r/'experiments/hc-fixed-gemm-20260913'
shutil.copy2(r/'deploy_config.json',e/'config.before.json')
c=json.loads((r/'deploy_config.json').read_text());assert not c.get('batch_invariant')
p=e/'hyperconnection.fixed.py';name='models/qwen4_exp/nvidia/hyperconnection.py'
c['runtime_overrides'][name]=str(p);c['runtime_override_sha256'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
(r/'deploy_config.json').write_text(json.dumps(c,indent=2)+'\n')
(e/'config.candidate.json').write_text(json.dumps(c,indent=2)+'\n')
