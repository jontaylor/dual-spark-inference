"""Prometheus interval summaries; ITL is a generation-burst interval with MTP."""
import collections,json,pathlib,re,sys
p=pathlib.Path(sys.argv[1])
def read(path):
 d=collections.defaultdict(float)
 for line in path.read_text().splitlines():
  if line.startswith('#'):continue
  m=re.match(r'([^ {]+)(?:\{(.*?)\})?\s+([-+0-9.eE]+)$',line)
  if not m:continue
  name,labels,value=m.groups();le=re.search(r'(?:^|,)le="([^"]+)"',labels or '')
  d[(name,le.group(1) if le else '')]+=float(value)
 return d
before=read(p/'metrics-before.txt');after=read(p/'metrics-after.txt');dt=json.loads((p/'capture.json').read_text());seconds=dt['end']-dt['start'];out={'seconds':seconds}
for prefix in ('vllm:time_to_first_token_seconds','vllm:inter_token_latency_seconds','vllm:e2e_request_latency_seconds','vllm:request_time_per_output_token_seconds'):
 n=after[(prefix+'_count','')]-before[(prefix+'_count','')];total=after[(prefix+'_sum','')]-before[(prefix+'_sum','')]
 if n<=0:continue
 z={'n':n,'mean_seconds':total/n};buckets=sorted((float(le),v-before[(name,le)]) for (name,le),v in after.items() if name==prefix+'_bucket')
 for q in (.5,.95):
  lower=0
  for upper,value in buckets:
   if value>=n*q:z[f'p{int(q*100)}_bucket_seconds']=[lower,upper];break
   lower=upper
 out[prefix]=z
for metric in ('vllm:generation_tokens_total','vllm:prompt_tokens_total'):
 out[metric+'_per_second']=(after[(metric,'')]-before[(metric,'')])/seconds
out['note']='Interval deltas; MTP inter_token_latency is time between generation outputs/bursts, not latency per individual token. Workload changes confound cross-window comparisons.'
(p/'metrics-summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
