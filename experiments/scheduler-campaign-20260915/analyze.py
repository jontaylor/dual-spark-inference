import gzip,json,pathlib,statistics,collections,csv,datetime,tarfile
from zoneinfo import ZoneInfo
P=pathlib.Path(__file__).resolve().parent

def aggregate(m):
 d=collections.defaultdict(float)
 for k,v in m.items():d[k.split('{')[0]]+=v
 return dict(d)
def summarize(rows,start,end):
 selected=[r for r in rows if start<=r['time']<=end]
 if len(selected)<2:return None
 a,b=selected[0],selected[-1];duration=b['time']-a['time'];ma,mb=aggregate(a['metrics']),aggregate(b['metrics'])
 deltas={k:mb[k]-v for k,v in ma.items() if k in mb and (k.endswith('_total') or k.endswith('_sum') or k.endswith('_count'))}
 gauges={k:{'mean':statistics.mean(aggregate(r['metrics']).get(k,0) for r in selected),'max':max(aggregate(r['metrics']).get(k,0) for r in selected)} for k in ma if any(x in k for x in ('num_requests_running','num_requests_waiting','kv_cache_usage_perc'))}
 label_deltas={k:b['metrics'][k]-v for k,v in a['metrics'].items() if k in b['metrics'] and k.split('{')[0].endswith('_total')}
 def ratio(n,d):return deltas.get(n,0)/deltas[d] if deltas.get(d,0)>0 else None
 return {'sampled_seconds':duration,'deltas':deltas,'label_deltas':label_deltas,'gauges':gauges,'generation_tok_s':deltas.get('vllm:generation_tokens_total',0)/duration,'prompt_tok_s':deltas.get('vllm:prompt_tokens_total',0)/duration,'cached_prompt_fraction':ratio('vllm:prompt_tokens_cached_total','vllm:prompt_tokens_total'),'mean_ttft_s':ratio('vllm:time_to_first_token_seconds_sum','vllm:time_to_first_token_seconds_count'),'mean_completed_request_latency_s':ratio('vllm:e2e_request_latency_seconds_sum','vllm:e2e_request_latency_seconds_count')}

results=[]
for arm in sorted(P.glob('[0-9][0-9]-*')):
 if not (arm/'complete.json').exists():continue
 w=json.loads((arm/'window.json').read_text());rows=[json.loads(l) for l in gzip.open(arm/'metrics.jsonl.gz','rt')]
 result={'arm':arm.name,'window':w,'complete':json.loads((arm/'complete.json').read_text()),'full':summarize(rows,w['start'],w['start']+600),'last_five':summarize(rows,w['start']+300,w['start']+600),'cutoff':json.loads((arm/'cutoff.json').read_text())}
 hist=collections.Counter();errors=[]
 for l in gzip.open(arm/'server-r0.log.gz','rt'):
  if 'ERROR' in l or 'Traceback' in l:errors.append(l.strip())
  if 'GB10_KV_ACCOUNTING {' in l:
   try:
    d=json.loads(l.split('GB10_KV_ACCOUNTING ',1)[1])
    if not w['start']<=d['time']<=w['start']+600:continue
    for r in d['requests'].values():
     n=r.get('in_flight',0)
     if n>4 and r.get('computed',0)<=r.get('prompt',0):hist[n]+=1
   except Exception:pass
 result['sampled_prefill_chunks']=dict(hist);result['server_errors']=errors
 hardware={}
 for l in gzip.open(arm/'hardware.jsonl.gz','rt'):
  r=json.loads(l)
  if 'data' not in r or not w['start']<=r['time']<=w['start']+600:continue
  try:
   v=[float(x.strip()) for x in r['data']['gpu'].strip().split(',')];hardware.setdefault(r['rank'],[]).append(v)
  except ValueError:pass
 result['gpu']={r:{'samples':len(vs),'mean_util_percent':statistics.mean(v[0] for v in vs),'single_digit_util_fraction':sum(v[0]<10 for v in vs)/len(vs),'mean_power_w':statistics.mean(v[2] for v in vs)} for r,vs in hardware.items()}
 dense_gpu={}
 for rank in (0,1):
  path=P/f'gpu-continuous-r{rank}.csv';vs=[]
  if path.exists():
   for row in csv.reader(path.open()):
    try:
     t=datetime.datetime.strptime(row[0].strip(),'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=ZoneInfo('Europe/London')).timestamp()
     if w['start']<=t<=w['start']+600:vs.append([float(x.strip()) for x in row[1:]])
    except (ValueError,IndexError):pass
  if vs:dense_gpu[rank]={'samples':len(vs),'mean_util_percent':statistics.mean(v[0] for v in vs),'single_digit_util_fraction':sum(v[0]<10 for v in vs)/len(vs),'mean_power_w':statistics.mean(v[2] for v in vs)}
 result['gpu_half_second']=dense_gpu
 campaign_runs=[]
 with tarfile.open(arm/'campaign-evidence.tar.gz') as archive:
  for member in archive:
   if member.name.count('/')==1 and member.name.endswith('/run.json'):
    data=json.load(archive.extractfile(member));calls=data.get('calls',[]);events=data.get('tool_events',[])
    campaign_runs.append({'arm':member.name.split('/')[0],'status':data.get('status','UNKNOWN'),'recorded_calls':len(calls) if isinstance(calls,list) else None,'recorded_tool_events':len(events) if isinstance(events,list) else None})
 result['campaign_runs_at_export']=campaign_runs
 result['profile_sampling_logs']={f.name:f.read_text() for f in arm.glob('profile-*.log')}
 (arm/'summary.json').write_text(json.dumps(result,indent=2));results.append(result)
(P/'results.json').write_text(json.dumps(results,indent=2))
lines=['# Scheduler campaign results','', ('All eight ten-minute arms are complete.' if len(results)==8 else 'Partial results; only completed ten-minute arms appear.') + ' Throughputs use counter deltas and include unfinished responses. Prompt throughput includes cached tokens. No full determinism claim.','', '| Arm | Generation tok/s · full | Generation tok/s · final 5m | Prompt tok/s · full | Same campaign throughout |','|---|---:|---:|---:|---|']
for r in results:
 lines.append(f"| {r['arm']} | {r['full']['generation_tok_s']:.2f} | {r['last_five']['generation_tok_s']:.2f} | {r['full']['prompt_tok_s']:.2f} | {r['complete']['same_campaign']} |")
lines+=['','| Arm | Cached prompt % | Mean TTFT · seconds | Completed requests | Running / waiting at cutoff |','|---|---:|---:|---:|---:|']
for r in results:
 f=r['full'];m=aggregate(r['cutoff']['metrics']);lines.append(f"| {r['arm']} | {100*f['cached_prompt_fraction']:.2f} | {f['mean_ttft_s']:.2f} | {f['deltas'].get('vllm:request_success_total',0):.0f} | {m.get('vllm:num_requests_running',0):.0f} / {m.get('vllm:num_requests_waiting',0):.0f} |")
lines+=['','Campaign run statuses and recorded call/tool counts are preserved in results.json. Completed inference requests are not completed benchmark tasks.', '', 'Latency averages cover observations recorded during the window, not a fixed cohort. External-cache source labels include the custom GPU-resident completion restore path and must not be equated with disk traffic.','', 'Raw counters, histogram sums/counts, sampled scheduler allocations, GPU summaries and cutoff request counts are in results.json. Campaign artifacts are in each arm’s campaign-evidence.tar.gz. Five-second GPU samples cannot measure every brief stall. Python stack samples are not GPU kernel traces.']
(P/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
