from pathlib import Path
import subprocess,json
p=Path(__file__).resolve().parent;py=str(p.parents[1]/'.venv/bin/python')
for script,mode in [('probe.py','full-spec'),('probe.py','full-spec-seed'),('uncached_probe.py','uncached-spec'),('uncached_probe.py','uncached-spec-seed')]:
 subprocess.run([py,str(p/script),mode],check=True)
 rows=json.loads((p/mode/'results.json').read_text());ref=rows[0]
 assert len(rows)==10 and ref['token_ids'] and ref['logprobs']
 assert all(r['error'] is None and r['prompt_token_ids']==ref['prompt_token_ids'] and r['token_ids']==ref['token_ids'] and r['logprobs']==ref['logprobs'] for r in rows),mode+' diverged'
 if mode.startswith('uncached'):assert all(r['usage']['prompt_tokens_details']['cached_tokens']==0 for r in rows)
 print(mode+' exact token/score equality passed',flush=True)
subprocess.run([py,str(p/'mixed_probe.py')],check=True)
