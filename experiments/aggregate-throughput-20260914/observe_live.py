import json,time,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent
prefixes=('vllm:prompt_tokens_total{','vllm:prompt_tokens_cached_total{','vllm:prompt_tokens_by_source_total{','vllm:generation_tokens_total{','vllm:prefix_cache_queries_total{','vllm:prefix_cache_hits_total{','vllm:external_prefix_cache_queries_total{','vllm:external_prefix_cache_hits_total{','vllm:num_requests_running{','vllm:num_requests_waiting{','vllm:request_success_total{','vllm:spec_decode_num_accepted_tokens_total{','vllm:spec_decode_num_draft_tokens_total{')
def parse(raw):return {l.rsplit(' ',1)[0]:float(l.rsplit(' ',1)[1]) for l in raw.splitlines() if l.startswith(prefixes)}
baseline=parse((p/'pre-resume-metrics.txt').read_text());start=json.loads((p/'campaign-resume-v2.json').read_text())['time']
with (p/'post-resume-metrics.jsonl').open('w') as f:
 for i in range(31):
  with urllib.request.urlopen('http://127.0.0.1:30001/metrics',timeout=5) as r:raw=r.read().decode()
  now=time.time(); current=parse(raw);delta={k:v-baseline.get(k,0) for k,v in current.items()};row={'time':now,'elapsed':now-start,'current':current,'delta':delta};f.write(json.dumps(row)+'\n');f.flush()
  def total(prefix):return sum(v for k,v in delta.items() if k.startswith(prefix))
  prompt=total('vllm:prompt_tokens_total{');cached=total('vllm:prompt_tokens_cached_total{');gen=total('vllm:generation_tokens_total{')
  print(json.dumps({'seconds':round(now-start),'prompt':prompt,'cached':cached,'cache_fraction':cached/prompt if prompt else None,'generated':gen,'aggregate_gen_tps':gen/(now-start),'completed':total('vllm:request_success_total{'),'running':sum(v for k,v in current.items() if k.startswith('vllm:num_requests_running{'))}),flush=True)
  if i<30:time.sleep(10)
(p/'post-resume-last-metrics.txt').write_text(raw)
