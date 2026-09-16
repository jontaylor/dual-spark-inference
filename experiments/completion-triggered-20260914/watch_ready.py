import time,subprocess,urllib.request,json
from pathlib import Path
p=Path(__file__).resolve().parent
start=time.time()
while time.time()-start<900:
 try:
  with urllib.request.urlopen('http://127.0.0.1:30001/health',timeout=2) as r:
   if r.status==200:
    print('READY',flush=True);break
 except Exception:pass
 r=subprocess.run(['docker','logs','--tail','12','qwen38-kv-paging-r0'],capture_output=True,text=True)
 logs=r.stdout+r.stderr
 if any(x in logs for x in ['Engine core initialization failed','torch.OutOfMemoryError','Traceback (most recent call last)']):raise RuntimeError(logs)
 print('loading',round(time.time()-start),logs[-350:],flush=True);time.sleep(20)
else:raise TimeoutError('Readiness deadline exceeded')
