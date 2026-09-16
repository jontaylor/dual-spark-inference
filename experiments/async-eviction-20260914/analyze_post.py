import json,datetime,collections
from pathlib import Path
from zoneinfo import ZoneInfo
p=Path(__file__).parent;q=p/'post-stalls';result={}
for rank in [0,1]:
 rows=[]
 for line in (q/f'gpu-r{rank}.csv').read_text().splitlines():
  a=line.split(', ')
  try:rows.append((datetime.datetime.strptime(a[0],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=ZoneInfo('Europe/London')).timestamp(),float(a[1]),float(a[3])))
  except (ValueError,IndexError):continue
 groups=[]
 for t,u,w in rows:
  if u<10:
   if not groups or t-groups[-1][-1]>.4:groups.append([])
   groups[-1].append(t)
 result[f'gpu{rank}']={'samples':len(rows),'low_samples':sum(u<10 for _,u,_ in rows),'low_episodes':len(groups),'mean_power':sum(w for _,_,w in rows)/len(rows),'min_util':min(u for _,u,_ in rows)}
d=json.loads((q/'worker.speedscope.json').read_text());f=d['shared']['frames'];main=d['profiles'][0];duration=sum(main['weights']);source=full=capture=0
for ids,w in zip(main['samples'],main['weights']):
 names=[f[i]['name'] for i in ids]
 if 'wait_source_preserved'in names:source+=w
 elif 'handle_preemptions'in names:full+=w
 if 'capture_gpu_completions'in names:capture+=w
result['profile']={'sampled_seconds':duration,'source_fence_seconds':source,'other_preemption_seconds':full,'source_fence_percent':100*source/duration,'capture_seconds':capture}
a=[json.loads(l) for l in (q/'metrics.jsonl').read_text().splitlines()];a=[r for r in a if 'metrics'in r];x,y=a[0],a[-1];dt=y['time']-x['time']
result['metrics']={'seconds':dt,'generation_tps':(y['metrics']['vllm:generation_tokens_total']-x['metrics']['vllm:generation_tokens_total'])/dt,'mean_running':sum(r['metrics']['vllm:num_requests_running'] for r in a)/len(a),'min_running':min(r['metrics']['vllm:num_requests_running'] for r in a),'max_running':max(r['metrics']['vllm:num_requests_running'] for r in a),'delta':{k:y['metrics'][k]-x['metrics'][k] for k in x['metrics'] if 'total'in k}}
(q/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
