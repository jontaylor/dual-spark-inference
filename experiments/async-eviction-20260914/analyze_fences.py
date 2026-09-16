import re,json,datetime,statistics
from pathlib import Path
p=Path(__file__).parent;allrows=[]
for rank in [0,1]:
 staged={};persisted={};rows=[]
 for l in (p/f'runtime-r{rank}.log').read_text().splitlines():
  try:t=datetime.datetime.fromisoformat(l.split()[0].replace('Z','+00:00')).timestamp()
  except (ValueError,IndexError):continue
  m=re.search(r'GB10_EVICTION_SOURCE_PRESERVED rank=\d+ job=(\d+) seconds=([\d.]+)',l)
  if m:staged[int(m[1])]=(t,float(m[2]))
  if 'GB10_DISK_TRANSFER {' in l:
   d=json.loads(l.split('GB10_DISK_TRANSFER ',1)[1])
   if d.get('source_staged'):persisted[d['job']]=(t,d)
 for jid,(t,source_elapsed) in staged.items():
  if jid in persisted:
   end,d=persisted[jid];rows.append({'job':jid,'preserved':t,'persisted':end,'lead_seconds':end-t,'source_queue_copy_seconds':source_elapsed,'total_seconds':d['seconds'],'payload_bytes':d['bytes']})
 s={'rank':rank,'source_events':len(staged),'completed_staged_jobs':len(rows),'min_lead_seconds':min((r['lead_seconds'] for r in rows),default=None),'median_lead_seconds':statistics.median(r['lead_seconds'] for r in rows) if rows else None,'max_lead_seconds':max((r['lead_seconds'] for r in rows),default=None),'rows':rows};allrows.append(s)
(p/'fence-evidence.json').write_text(json.dumps(allrows,indent=2));print(json.dumps([{k:v for k,v in x.items() if k!='rows'} for x in allrows],indent=2))
