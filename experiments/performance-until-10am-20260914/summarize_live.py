import json,time,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent
r=json.loads((p/'content-validation-resume.json').read_text());start=r['time']
def parse(raw):return {l.rsplit(' ',1)[0]:float(l.rsplit(' ',1)[1]) for l in raw.splitlines() if l.startswith('vllm:') and '_bucket{' not in l and '_created{' not in l and 'config_info{' not in l}
base=parse((p/'content-pre-resume-metrics.txt').read_text());rows=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()];latest=next(x for x in reversed(rows) if x.get('metrics'));delta={k:v-base.get(k,0) for k,v in latest['metrics'].items()}
def total(name):return sum(v for k,v in delta.items() if k.startswith('vllm:'+name+'{'))
def current(name):return sum(v for k,v in latest['metrics'].items() if k.startswith('vllm:'+name+'{'))
secs=latest['time']-start;prompt=total('prompt_tokens_total');cached=total('prompt_tokens_cached_total')
s={'start':start,'sample_time':latest['time'],'seconds':secs,'metric_age_seconds':time.time()-latest['time'],'config_sha256':latest['config_sha256'],'running':current('num_requests_running'),'waiting':current('num_requests_waiting'),'generated_tokens':total('generation_tokens_total'),'aggregate_generated_tps':total('generation_tokens_total')/secs,'prompt_tokens':prompt,'cached_tokens':cached,'cache_fraction':cached/prompt if prompt else None,'completed_requests':total('request_success_total'),'store_payload_bytes':total('kv_offload_store_bytes_total'),'load_payload_bytes':total('kv_offload_load_bytes_total'),'preemptions':total('num_preemptions_total')}
assert min(s['generated_tokens'],s['prompt_tokens'],s['store_payload_bytes'],s['load_payload_bytes'])>=0,'Counter reset: split epoch before comparing'
(p/'content-live-interval.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
