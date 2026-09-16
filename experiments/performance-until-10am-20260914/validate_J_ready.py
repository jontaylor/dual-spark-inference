import json,subprocess,time,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent;start=time.monotonic();last=0
while time.monotonic()-start<900:
 state=subprocess.check_output(['systemctl','show','qwen38-next-qwen-fp8.service','--property=ActiveState','--value'],text=True,timeout=5).strip()
 if state in ('failed','inactive'):raise RuntimeError('Host service is '+state+'; inspect journal before further waiting')
 try:
  with urllib.request.urlopen('http://127.0.0.1:30001/health',timeout=2) as r:
   if r.status==200:break
 except Exception:pass
 logs=subprocess.run(['docker','logs','--tail','60','qwen38-kv-paging-r0'],capture_output=True,text=True);text=logs.stdout+logs.stderr
 if any(marker in text for marker in ('Engine core initialization failed','Error response from daemon:','torch.OutOfMemoryError','RuntimeError: Failed to')):
  if 'No such container' not in text:raise RuntimeError('Known startup error: '+text[-5000:])
 if time.monotonic()-last>60:print('Waiting for J readiness; elapsed',round(time.monotonic()-start),flush=True);last=time.monotonic()
 time.sleep(5)
else:raise TimeoutError('Observation deadline; inspect actual process, do not infer terminal state')
import hashlib
assert hashlib.sha256((p.parents[1]/'deploy_config.json').read_bytes()).hexdigest()=='7dadca0c648ba60cf978654d2f6a4a19f40b5ee6a7ad456985284586272ab845'
r=subprocess.run(['docker','logs','qwen38-kv-paging-r0'],capture_output=True,text=True,check=True)
assert 'unit_tokens=7680 unit_blocks=4 per_request_blocks=34' in r.stdout+r.stderr, 'Expected reservation geometry not observed'
subprocess.run(['python3',str(p/'verify_J_runtime.py')],check=True)
print('J ready and reservation geometry verified; running live cache/full/seed/mixed gates',flush=True)
subprocess.run([str(p.parents[1]/'.venv/bin/python'),str(p/'validate_candidate.py'),'J'],check=True)
print('J primary gates passed. Workload remains paused for remaining review/disk checks.',flush=True)
