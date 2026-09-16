"""Matched elapsed workload counters, descriptive rather than causal comparison."""
import json,time,statistics
from pathlib import Path
p=Path(__file__).resolve().parent
cycles=[json.loads((p/f'batch-{i:03d}-counters.json').read_text()) for i in (2,3)]
allrows=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()]
duration=cycles[1]['seconds'];out=[]
for label,c in zip(['H','I3'],cycles):
 rows=[r for r in allrows if r.get('metrics') and r['config_sha256']==c['config_sha256'] and c['counter_start']<=r['time']<=c['counter_start']+duration+.1]
 a,b=rows[0],rows[-1]
 def val(r,n):return sum(v for k,v in r['metrics'].items() if k.startswith('vllm:'+n+'{'))
 def d(n):return val(b,n)-val(a,n)
 seconds=b['time']-a['time'];prompt=d('prompt_tokens_total');drafts=d('spec_decode_num_drafts_total')
 out.append({'config':label,'start':a['time'],'end':b['time'],'seconds':seconds,'aggregate_tps':d('generation_tokens_total')/seconds,'generated_tokens':d('generation_tokens_total'),'cache_fraction':d('prompt_tokens_cached_total')/prompt if prompt else None,'uncached_prompt_tokens':prompt-d('prompt_tokens_cached_total'),'completed_requests':d('request_success_total'),'accepted_drafts_per_attempt':d('spec_decode_num_accepted_tokens_total')/drafts if drafts else None,'mean_running_samples':statistics.mean(val(r,'num_requests_running') for r in rows),'preemptions':d('num_preemptions_total'),'payload_written':d('kv_offload_store_bytes_total'),'payload_loaded':d('kv_offload_load_bytes_total')})
s={'time':time.time(),'rows':out,'scope':'Same elapsed start of frozen-workload cycles; different model outputs, context lengths, concurrency and cache state. H includes paired readback windows and probes after about ten minutes. I3 changes draft depth and page layout. Not isolated causal speedup or final cycle comparison.'}
(p/'H-I3-matched-elapsed.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
