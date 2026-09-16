import json,hashlib,subprocess,shlex
from pathlib import Path
root=Path('/home/jon/dual-spark-inference-kv-paging');e=root/'experiments/targeted-determinism-20260913'
p=e/'model.final.py';key='models/qwen4_exp/nvidia/model.py'
for rank in (0,1):
 cfg=json.loads((e/f'candidate-dense-r{rank}.json').read_text())
 cfg['runtime_overrides'][key]=str(p);cfg['runtime_override_sha256'][key]=hashlib.sha256(p.read_bytes()).hexdigest()
 (e/f'candidate-final-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
subprocess.run(['scp',str(p),'192.168.100.11:'+str(p)],check=True)
subprocess.run(['scp',str(e/'candidate-final-r1.json'),'192.168.100.11:'+str(root/'deploy_config.json')],check=True)
(root/'deploy_config.json').write_bytes((e/'candidate-final-r0.json').read_bytes())
message='Prospective scalar-gate fix passes complete117-token serial3/concurrent4/serial3 with exact token and score equality. Deploying same verified fix permanently, then validating normal CUDA graphs and C8 mixed prompts. Concurrency is retained; preserve representative workload settings and retries.'
subprocess.run(['ssh','jon@192.168.0.167',shlex.join(['codex','queue','--thread','01a08cc2-ed05-7073-aa5a-7bebae318c0e','--message',message])],check=True,timeout=30)
