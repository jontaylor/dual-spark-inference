import json,subprocess,time,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent;start=time.monotonic();last=0
while time.monotonic()-start<900:
 try:
  with urllib.request.urlopen('http://127.0.0.1:30001/health',timeout=2) as r:
   if r.status==200:break
 except Exception:pass
 logs=subprocess.run(['docker','logs','--tail','60','qwen38-kv-paging-r0'],capture_output=True,text=True);text=logs.stdout+logs.stderr
 if any(marker in text for marker in ('Engine core initialization failed','Error response from daemon:','torch.OutOfMemoryError','RuntimeError: Failed to')):
  if 'No such container' not in text:raise RuntimeError('Known startup error: '+text[-5000:])
 if time.monotonic()-last>60:print('Waiting for I3 readiness; elapsed',round(time.monotonic()-start),flush=True);last=time.monotonic()
 time.sleep(5)
else:raise TimeoutError('Observation deadline; inspect actual process, do not infer terminal state')
print('I3 ready; running live cache/full/seed/mixed gates',flush=True)
subprocess.run([str(p.parents[1]/'.venv/bin/python'),str(p/'validate_candidate.py'),'I3'],check=True)
print('I3 primary gates passed. Workload remains paused for remaining review/disk checks.',flush=True)
