from pathlib import Path
import subprocess,json,shlex,hashlib
root=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent
m='Aligned cache reuse works:7360/7393 tokens reused vs4800 before. C8 testing caught a boundary-clipping regression in first candidate; reverting scheduler byte-for-byte to previously validated MTP scheduler. Revised capture takes the already-retained accepted state after crossing each64-token boundary, without changing verification batch lengths; crossing1/2/3-token state tests pass. Deploying this correction now. Campaign processes remain paused and will be resumed after short validation. Do not change workload sampling/retries.'
subprocess.run(['ssh','jon@192.168.0.167',shlex.join(['codex','queue','--thread','01a09b9f-c6c7-7bb0-8c25-b22748f43388','--message',m])],check=True,timeout=30)
subprocess.run(['ssh','192.168.100.11','mkdir','-p',str(p)],check=True)
for name in ['completion.aligned.py','connector.aligned.py','scheduler.aligned.py']:
 f=p/name;compile(f.read_text(),str(f),'exec');subprocess.run(['scp',str(f),'192.168.100.11:'+str(f)],check=True)
for rank in [0,1]:
 cfg=json.loads((p/f'candidate-aligned-r{rank}.json').read_text())
 for k,f in cfg['runtime_overrides'].items():assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==cfg['runtime_override_sha256'][k]
(root/'deploy_config.json').write_bytes((p/'candidate-aligned-r0.json').read_bytes())
subprocess.run(['scp',str(p/'candidate-aligned-r1.json'),'192.168.100.11:'+str(root/'deploy_config.json')],check=True)
subprocess.run(['sudo','-n','systemctl','restart','--no-block','qwen38-next-qwen-fp8.service'],check=True)
print('Verified candidate deployed; nonblocking managed restart requested.')
