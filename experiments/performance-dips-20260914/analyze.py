import re,json,statistics
from pathlib import Path
p=Path(__file__).resolve().parent;lines=(p/'server.log').read_text().splitlines();snaps=[];rates=[];spec=[]
def stamp(s):
 m=re.search(r'INFO \d+-\d+ (\d+):(\d+):(\d+)',s)
 return sum(int(v)*w for v,w in zip(m.groups(),[3600,60,1])) if m else None
for l in lines:
 if 'GB10_KV_ACCOUNTING {' in l:
  d=json.loads(l.split('GB10_KV_ACCOUNTING ',1)[1]);req=list(d['requests'].values());snaps.append({'t':stamp(l),'prefills':sum(r['computed']<r['prompt'] for r in req),'prefill_in_flight':sum(r['in_flight'] for r in req if r['computed']<r['prompt']),'requests':len(req),'snapshot':d})
 m=re.search(r'Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+)',l)
 if m:rates.append({'t':stamp(l),'prompt_tps':float(m[1]),'generation_tps':float(m[2]),'running':int(m[3]),'waiting':int(m[4])})
 m=re.search(r'Mean acceptance length: ([\d.]+).*Avg Draft acceptance rate: ([\d.]+)%',l)
 if m:spec.append({'t':stamp(l),'accept_length':float(m[1]),'accept_rate':float(m[2])})
for r in rates:
 ss=[s for s in snaps if r['t']-10<=s['t']<=r['t']]
 r['prefill_samples']=len(ss);r['mean_prefills']=round(statistics.mean(s['prefills'] for s in ss),2) if ss else None
 s=next((s for s in spec if s['t']==r['t']),{});r.update({k:v for k,v in s.items() if k!='t'})
print('Last 18 ten-second windows: UTC, generation/s, running, average prefills, acceptance%')
for r in rates[-18:]:print(f"{r['t']//3600:02}:{r['t']//60%60:02}:{r['t']%60:02}",r['generation_tps'],r['running'],r['mean_prefills'],r.get('accept_rate'))
bins={}
for label,pred in [('no_prefill',lambda n:n==0),('some_prefill',lambda n:n>0),('at_least_two_prefills',lambda n:n>=2)]:
 a=[r['generation_tps'] for r in rates if r['running']>0 and r['mean_prefills'] is not None and pred(r['mean_prefills'])]
 if a:bins[label]={'windows':len(a),'median_tps':statistics.median(a),'min_tps':min(a),'max_tps':max(a)}
print(json.dumps(bins,indent=2));(p/'analysis.json').write_text(json.dumps({'windows':rates,'groups':bins,'snapshots':snaps},indent=2)+'\n')
