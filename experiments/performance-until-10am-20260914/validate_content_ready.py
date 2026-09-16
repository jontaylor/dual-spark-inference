import subprocess,time,urllib.request,json
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1];py=str(root/'.venv/bin/python');start=time.monotonic();prior=p.parent/'aggregate-throughput-20260914';spec=p.parent/'spec-determinism-20260914'
while time.monotonic()-start<900:
 try:
  with urllib.request.urlopen('http://127.0.0.1:30001/health',timeout=2) as r:
   if r.status==200:break
 except Exception:pass
 logs=subprocess.run(['docker','logs','--tail','40','qwen38-kv-paging-r0'],capture_output=True,text=True)
 if 'Engine core initialization failed' in logs.stdout+logs.stderr:raise RuntimeError('Engine initialization failed; stop and diagnose')
 time.sleep(5)
else:raise TimeoutError('Readiness not achieved; inspect actual engine before deciding rollback')
print('Ready, checking cache and concurrent scores',flush=True)
subprocess.run([py,str(prior/'cache_probe.py'),'after-content'],check=True)
for mode in ['full-content','full-content-seed']:
 subprocess.run([py,str(spec/'probe.py'),mode],check=True)
 rows=json.loads((spec/mode/'results.json').read_text());ref=rows[0]
 assert len(rows)==10 and all(r['error'] is None and all(r[k]==ref[k] for k in ['prompt_token_ids','token_ids','logprobs']) for r in rows),mode+' divergence'
subprocess.run([py,str(spec/'mixed_probe.py')],check=True)
m=sorted(d for d in spec.glob('mixed-*') if d.is_dir())[-1];summary=json.loads((m/'summary.json').read_text());assert all(all(r[k] for k in ['tokens_equal','scores_equal','prompt_equal']) for r in summary)
(p/'content-model-validation.json').write_text(json.dumps({'normal_seed_requests':20,'exact_tokens_scores':True,'mixed':str(m),'mixed_all_equal':True},indent=2))
subprocess.run([py,str(p/'disk_probe.py')],check=True)
print('Content-dedup validation complete; inspect disk event totals and resume recorded workload.',flush=True)
