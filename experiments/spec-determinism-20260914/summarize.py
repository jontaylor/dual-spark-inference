import json,statistics,time,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent;old=p.parent/'targeted-determinism-20260913';modes=['smoke-spec','normal-spec','normal-spec-seed','full-spec','full-spec-seed','uncached-spec','uncached-spec-seed'];out=[];allrows=[]
full=json.loads((p/'full-spec/results.json').read_text())[0]
for mode in modes:
 rows=json.loads((p/mode/'results.json').read_text());assert len(rows)==10;ref=rows[0]
 assert all(r['error'] is None and r['prompt_token_ids']==ref['prompt_token_ids'] and r['token_ids']==ref['token_ids'] and r['logprobs']==ref['logprobs'] for r in rows)
 # Different generation limits may change finish records, but token/score prefixes must match.
 assert all(r['token_ids']==full['token_ids'][:len(r['token_ids'])] and r['logprobs']==full['logprobs'][:len(r['logprobs'])] for r in rows)
 if mode.startswith('uncached'):assert all(r['usage']['prompt_tokens_details']['cached_tokens']==0 for r in rows)
 serial=[r['seconds'] for r in rows if not r['name'].startswith('parallel')];parallel=[r for r in rows if r['name'].startswith('parallel')];wall=max(r['started']+r['seconds'] for r in parallel)-min(r['started'] for r in parallel)
 out.append(dict(mode=mode,requests=10,tokens_per_response=len(ref['token_ids']),exact_tokens_scores=True,serial_median_seconds=statistics.median(serial),C4_wall_seconds=wall,C4_aggregate_tokens_per_second=sum(len(r['token_ids']) for r in parallel)/wall,cached_tokens=sorted(set(r['usage']['prompt_tokens_details']['cached_tokens'] for r in rows))))
 allrows+=rows
mixed=sorted(d for d in p.glob('mixed-*') if d.is_dir())[-1];s=json.loads((mixed/'summary.json').read_text());assert len(s)==12 and all(r[k] for r in s for k in ['tokens_equal','scores_equal','prompt_equal'])
metrics=[json.loads(l) for l in (mixed/'metrics.jsonl').read_text().splitlines()];peak=max(sum(float(l.rsplit(' ',1)[1]) for l in r.get('metrics',[]) if l.startswith('vllm:num_requests_running{')) for r in metrics);assert peak==8
mr=[json.loads(f.read_text()) for f in mixed.glob('*.json') if f.name!='summary.json'];assert len(mr)==16
scores=[score for r in allrows for score in r['logprobs']]+[score for r in mr for score in r['scores']]
mismatch=sum(s['logprob']<max(t['logprob'] for t in s['top_logprobs']) for s in scores if s.get('top_logprobs'))
assert mismatch==0
result={'time':time.time(),'experiments':out,'mixed':{'directory':mixed.name,'requests':16,'peak_concurrency':peak,'prompt_lengths':sorted(set(len(r['prompt_ids']) for r in mr)),'exact_tokens_scores':True},'requests_total':len(allrows)+len(mr),'selected_tokens_checked':len(scores),'selection_score_mismatches':mismatch,'cross_limit_and_seed_token_score_prefix_equality':True,'cross_spec_off_equivalence':False}
(p/'validation-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
with urllib.request.urlopen('http://127.0.0.1:30001/metrics',timeout=5) as r:(p/'metrics-final.txt').write_bytes(r.read())
