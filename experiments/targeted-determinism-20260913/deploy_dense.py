import json,hashlib,subprocess,shlex
from pathlib import Path
root=Path('/home/jon/dual-spark-inference-kv-paging');e=root/'experiments/targeted-determinism-20260913'
mapping={'model_executor/determinism/gb10_targeted.py':e/'targeted_kernels.dense.py','models/qwen4_exp/nvidia/model.py':e/'model.fixed.py','models/qwen4_exp/nvidia/ple_layer.py':e/'ple_layer.fixed.py','v1/worker/gpu/gb10_row_trace.py':e/'row_trace.dense.py'}
for rank in (0,1):
 cfg=json.loads((e/f'candidate-deep-r{rank}.json').read_text())
 cfg['runtime_overrides'].update({k:str(v) for k,v in mapping.items()});cfg['runtime_override_sha256'].update({k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in mapping.items()})
 (e/f'candidate-dense-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
for p in mapping.values():subprocess.run(['scp',str(p),'192.168.100.11:'+str(p)],check=True)
subprocess.run(['scp',str(e/'candidate-dense-r1.json'),'192.168.100.11:'+str(root/'deploy_config.json')],check=True)
(root/'deploy_config.json').write_bytes((e/'candidate-dense-r0.json').read_bytes())
message='All-row/eager trace isolated additional BF16 router, shared-expert and PLE projection differences plus final head logits with equal hidden states. Primitive-tested targeted fixes now deploying; no serialization, normal CUDA graphs retained. Final-head fixed kernel ~2.8-2.9ms isolated, near stock C4/C8. Preserve workload cadence/retries/settings.'
subprocess.run(['ssh','jon@192.168.0.167',shlex.join(['codex','queue','--thread','01a08cc2-ed05-7073-aa5a-7bebae318c0e','--message',message])],check=True,timeout=30)
