import json,shlex,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];e=Path(__file__).resolve().parent
message='Preparing a validated three-token MTP candidate to restore speed while preserving deterministic target inference. Short GB10 tests pass output/state equality for C1/C4, ordinary vs speculative recurrence, all1-4 acceptance/rejection resumes, and mixed prefill. Restarting managed inference service for whole-model validation. Preserve representative workload sampling, cadence and retries; retain raw evidence for any errors/divergence. Prior deterministic no-spec config is archived for rollback.'
subprocess.run(['ssh','jon@192.168.0.167',shlex.join(['codex','queue','--thread','01a08cc2-ed05-7073-aa5a-7bebae318c0e','--message',message])],check=True,timeout=30)
subprocess.run(['ssh','192.168.100.11','mkdir','-p',str(e)],check=True)
for name in ['gdn.spec.py','scheduler.spec.py','fused_sigmoid.fixed.py']:
 subprocess.run(['scp',str(e/name),'192.168.100.11:'+str(e/name)],check=True)
(root/'deploy_config.json').write_bytes((e/'candidate-r0.json').read_bytes())
subprocess.run(['scp',str(e/'candidate-r1.json'),'192.168.100.11:'+str(root/'deploy_config.json')],check=True)
for rank in [0,1]:
 args=[str(root/'.venv/bin/python'),str(root/'launch_rank.py'),str(rank),'--dry-run']
 # launcher refuses existing running containers only outside dry-run.
 if rank:args=['ssh','192.168.100.11',shlex.join(args)]
 result=subprocess.run(args,capture_output=True,text=True,timeout=60)
 (e/f'preflight-r{rank}.log').write_text(result.stdout+result.stderr)
 if result.returncode:raise RuntimeError(f'Preflight rank{rank} failed; inspect log before restarting')
subprocess.run(['sudo','-n','systemctl','restart','qwen38-next-qwen-fp8.service'],check=True,timeout=60)
print('Both preflights passed; managed restart requested.')
