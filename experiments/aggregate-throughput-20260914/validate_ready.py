import time,subprocess,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent;start=time.monotonic()
while time.monotonic()-start<900:
 try:
  with urllib.request.urlopen('http://127.0.0.1:30001/health',timeout=2) as r:
   if r.status==200:break
 except Exception:pass
 logs=subprocess.run(['docker','logs','--tail','50','qwen38-kv-paging-r0'],capture_output=True,text=True)
 if 'Engine core initialization failed' in logs.stdout+logs.stderr:raise RuntimeError('Engine initialization failed')
 time.sleep(5)
else:raise TimeoutError('Readiness deadline')
print('Ready; running exact continuation cache probe',flush=True)
subprocess.run([str(p.parents[1]/'.venv/bin/python'),str(p/'cache_probe.py'),'after-cache-v2'],check=True)

subprocess.run([str(p.parents[1]/".venv/bin/python"),str(p/"validate_model.py")],check=True)
subprocess.run([str(p.parents[1]/".venv/bin/python"),str(p/"check_continuation.py")],check=True)

subprocess.run([str(p.parents[1]/".venv/bin/python"),str(p/"benchmark_reuse.py")],check=True)
