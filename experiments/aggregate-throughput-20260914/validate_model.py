import json,subprocess,statistics,sys
from pathlib import Path
p=Path(__file__).resolve().parent;old=p.parent/'spec-determinism-20260914';py=str(p.parents[1]/'.venv/bin/python');reports=[]
for mode in ['full-aligned-v2','full-aligned-v2-seed']:
 subprocess.run([py,str(old/'probe.py'),mode],check=True)
 rows=json.loads((old/mode/'results.json').read_text());ref=rows[0];assert len(rows)==10 and ref['token_ids'] and ref['logprobs']
 assert all(r['error'] is None and r['prompt_token_ids']==ref['prompt_token_ids'] and r['token_ids']==ref['token_ids'] and r['logprobs']==ref['logprobs'] for r in rows),mode+' divergence'
 previous=json.loads((old/'full-spec/results.json').read_text())[0]
 reports.append({'mode':mode,'requests':10,'tokens':len(ref['token_ids']),'exact_tokens_scores':True,'previous_tokens_equal':ref['token_ids']==previous['token_ids'],'previous_scores_equal':ref['logprobs']==previous['logprobs'],'serial_median_seconds':statistics.median(r['seconds'] for r in rows if not r['name'].startswith('parallel'))})
 print(reports[-1],flush=True)
(p/'model-validation.json').write_text(json.dumps(reports,indent=2)+'\n')
subprocess.run([py,str(old/'mixed_probe.py')],check=True)

mixed=sorted(d for d in old.glob("mixed-*") if d.is_dir())[-1]
summary=json.loads((mixed/"summary.json").read_text())
assert all(r["tokens_equal"] and r["scores_equal"] and r["prompt_equal"] for r in summary), "Mixed C8 divergence"
print("Mixed C8 exact tokens and scores passed",flush=True)
