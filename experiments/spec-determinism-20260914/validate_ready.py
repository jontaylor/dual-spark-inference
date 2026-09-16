import subprocess,time,urllib.request,json
from pathlib import Path
root=Path(__file__).resolve().parent;python=str(root.parents[1]/'.venv/bin/python');start=time.monotonic()
while time.monotonic()-start<900:
 try:
  with urllib.request.urlopen('http://127.0.0.1:30001/health',timeout=2) as r:
   if r.status==200:break
 except Exception:pass
 logs=subprocess.run(['docker','logs','--tail','40','qwen38-kv-paging-r0'],capture_output=True,text=True)
 if 'Engine core initialization failed' in logs.stdout+logs.stderr:raise RuntimeError('Engine failed; inspect logs')
 time.sleep(5)
else:raise TimeoutError('Readiness deadline exceeded')
print('API healthy; smoke probe',flush=True)
for mode in ['smoke-spec','normal-spec','normal-spec-seed']:
 subprocess.run([python,str(root/'probe.py'),mode],check=True)
 rows=json.loads((root/mode/'results.json').read_text());ref=rows[0]
 assert len(rows)==10 and ref['token_ids'] and ref['logprobs']
 assert all(r['error'] is None and r['prompt_token_ids']==ref['prompt_token_ids'] and r['token_ids']==ref['token_ids'] and r['logprobs']==ref['logprobs'] for r in rows),mode+' diverged'
 print(mode+' exact tokens/scores passed',flush=True)
