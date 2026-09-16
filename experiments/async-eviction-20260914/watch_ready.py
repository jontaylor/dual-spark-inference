import subprocess,time,urllib.request,json
from pathlib import Path
p=Path(__file__).parent;start=time.time();last=''
while time.time()-start<900:
 try:
  if urllib.request.urlopen('http://127.0.0.1:30001/health',timeout=3).status==200:
   print('READY',flush=True);(p/'ready.json').write_text(json.dumps({'epoch':time.time()}));break
 except Exception:pass
 state=subprocess.run(['systemctl','is-active','qwen38-next-qwen-fp8.service'],capture_output=True,text=True).stdout.strip()
 logs=subprocess.run(['docker','logs','--tail','12','qwen38-kv-paging-r0'],capture_output=True,text=True);s=logs.stdout+logs.stderr
 if state=='failed' or any(x in s for x in ['Engine core initialization failed','Traceback (most recent call last)','CUDA out of memory']):raise RuntimeError(state+' '+s[-2500:])
 lines=[l for l in s.splitlines() if 'INFO' in l or 'Loading' in l]
 line=lines[-1] if lines else state
 if line!=last:print(line,flush=True);last=line
 time.sleep(10)
else:raise TimeoutError('not ready within900s')
