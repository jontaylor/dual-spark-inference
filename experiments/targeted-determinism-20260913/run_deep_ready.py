import subprocess,time,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parent
start=time.monotonic()
while time.monotonic()-start<900:
    status=subprocess.run(['systemctl','is-active','qwen38-next-qwen-fp8.service'],capture_output=True,text=True)
    if status.stdout.strip()!='active':raise RuntimeError('Serving service stopped')
    logs=subprocess.run(['docker','logs','--tail','20','qwen38-kv-paging-r0'],capture_output=True,text=True)
    if 'Engine core initialization failed' in logs.stdout+logs.stderr:raise RuntimeError('Engine failed')
    try:
        with urllib.request.urlopen('http://127.0.0.1:30001/health',timeout=2) as r:
            if r.status==200:break
    except Exception:pass
    time.sleep(5)
else:raise TimeoutError('Candidate readiness deadline exceeded')
print('Diagnostic candidate healthy; running eager all-row trace',flush=True)
subprocess.run([str(root.parents[1]/'.venv/bin/python'),str(root/'deep_trace.py'),'captured-deep'],check=True)
