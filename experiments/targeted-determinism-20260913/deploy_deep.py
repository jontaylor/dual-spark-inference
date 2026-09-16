import json,hashlib,subprocess,shlex
from pathlib import Path
root=Path('/home/jon/dual-spark-inference-kv-paging');e=root/'experiments/targeted-determinism-20260913'
mapping={'v1/worker/gpu/gb10_row_trace.py':e/'row_trace.deep.py','v1/worker/gpu/model_runner.py':e/'model_runner.deep.py'}
for rank in (0,1):
 cfg=json.loads((e/f'candidate-qsa-r{rank}.json').read_text())
 cfg['runtime_overrides'].update({k:str(v) for k,v in mapping.items()});cfg['runtime_override_sha256'].update({k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in mapping.items()})
 (e/f'candidate-deep-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
for p in mapping.values():subprocess.run(['scp',str(p),'192.168.100.11:'+str(p)],check=True)
subprocess.run(['scp',str(e/'candidate-deep-r1.json'),'192.168.100.11:'+str(root/'deploy_config.json')],check=True)
(root/'deploy_config.json').write_bytes((e/'candidate-deep-r0.json').read_bytes())
message='Extended fixes move first differing prefill output from layer3 to GDN45. Mixed decode still differs. Adding opt-in eager decode/all-row diagnostics; normal serving keeps graphs and concurrency. Preserve representative workload settings/retries.'
subprocess.run(['ssh','jon@192.168.0.167',shlex.join(['codex','queue','--thread','01a08cc2-ed05-7073-aa5a-7bebae318c0e','--message',message])],check=True,timeout=30)
