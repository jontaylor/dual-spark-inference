import json,hashlib,subprocess,shlex
from pathlib import Path
root=Path('/home/jon/dual-spark-inference-kv-paging');e=root/'experiments/targeted-determinism-20260913'
mapping={'models/qwen4_exp/nvidia/qsa.py':e/'qsa.fixed.py','models/qwen4_exp/nvidia/ops/qsa.py':e/'qsa_ops.fixed.py','v1/worker/gpu/gb10_row_trace.py':e/'row_trace.qsa.py'}
for rank in (0,1):
 cfg=json.loads((e/f'candidate-r{rank}.json').read_text())
 cfg['runtime_overrides'].update({k:str(v) for k,v in mapping.items()});cfg['runtime_override_sha256'].update({k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in mapping.items()})
 (e/f'candidate-qsa-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
for p in mapping.values():subprocess.run(['scp',str(p),'192.168.100.11:'+str(p)],check=True)
subprocess.run(['scp',str(e/'candidate-qsa-r1.json'),'192.168.100.11:'+str(root/'deploy_config.json')],check=True)
(root/'deploy_config.json').write_bytes((e/'candidate-qsa-r0.json').read_bytes())
message='HC/GDN fixes pass captured live tensor equality through layers0 and2. Next source isolated in layer3 full attention: BF16 projections and M-dependent QSA split reduction. Deploying primitive-tested fixes to those now; no serialization. Preserve representative-load retries/settings.'
subprocess.run(['ssh','jon@192.168.0.167',shlex.join(['codex','queue','--thread','01a08cc2-ed05-7073-aa5a-7bebae318c0e','--message',message])],check=True,timeout=30)
