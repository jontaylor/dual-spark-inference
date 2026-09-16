import json,datetime,collections
from pathlib import Path
from zoneinfo import ZoneInfo
p=Path(__file__).parent
start=json.loads((p/'start.json').read_text())['epoch']
allg={};events=[]
for r in [0,1]:
 data=[]
 for l in (p/f'gpu-r{r}.csv').read_text().splitlines():
  a=l.split(', ')
  try:t=datetime.datetime.strptime(a[0],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=ZoneInfo('Europe/London')).timestamp();data.append({'time':t,'util':float(a[1]),'power':float(a[3])})
  except ValueError:continue
 allg[r]=data;groups=[]
 for d in data:
  if d['util']<10:
   if not groups or d['time']-groups[-1][-1]['time']>.4:groups.append([])
   groups[-1].append(d)
 for g in groups:events.append({'rank':r,'start':g[0]['time'],'end':g[-1]['time']+.2,'min_util':min(x['util'] for x in g),'min_power':min(x['power'] for x in g)})
profiles={}
for label in ['worker','engine','worker-r1']:
 path=p/f'{label}.speedscope.json'
 if not path.exists():continue
 d=json.loads(path.read_text());frames=d['shared']['frames'];ps=[]
 for prof in d['profiles']:
  if not prof.get('samples'):continue
  count=collections.Counter();t=0;timeline=[]
  for sample,w in zip(prof['samples'],prof['weights']):
   stack=[frames[i] for i in sample];count[stack[-1]['name'] if stack else 'EMPTY']+=w;timeline.append((t,t+w,stack));t+=w
  ps.append((prof['name'],timeline,count,t))
 profiles[label]=ps
 print(label,[(a,round(t,2),c.most_common(4)) for a,_,c,t in ps][:12])
print('EVENTS',json.dumps(events,indent=2));(p/'events.json').write_text(json.dumps(events,indent=2))
