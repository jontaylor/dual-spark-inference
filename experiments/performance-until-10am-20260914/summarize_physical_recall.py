"""Summarize retired new physical blobs; keep live and preexisting separate."""
import json,subprocess,statistics,time,sys
from pathlib import Path
p=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parent
reports=[]
for rank in (0,1):
 def read(name):
  if rank:return subprocess.check_output(['ssh','jon@192.168.100.11','cat',str(p/name)],text=True,timeout=20)
  return (p/name).read_text()
 summary=json.loads(read(f'physical-recall-r{rank}.json'))
 events=[json.loads(l) for l in read(f'physical-recall-r{rank}.jsonl').splitlines() if l]
 if summary['errors']:raise RuntimeError(summary['errors'])
 clean=[e for e in events if not e['preexisting'] and e['time']<=summary['time']]
 assert len(clean)==summary['cohorts']['new_retired']['generations'], 'Retirement snapshot mismatch'
 groups={}
 for group in sorted({e['name'].split('-',1)[0] for e in clean}):
  rows=[e for e in clean if e['name'].split('-',1)[0]==group];age=[e['time']-e['attached'] for e in rows]
  groups[group]={'generations':len(rows),'bytes':sum(e['size'] for e in rows),'unread_generations':sum(not e['observed_read'] for e in rows),'unread_bytes':sum(e['size'] for e in rows if not e['observed_read']),'median_lifetime_seconds':statistics.median(age),'min_lifetime_seconds':min(age),'max_lifetime_seconds':max(age)}
 reports.append({'rank':rank,'started':summary['started'],'sample_time':summary['time'],'age_seconds':time.time()-summary['time'],'cohorts':summary['cohorts'],'new_retired_by_group':groups})
result={'scope':'New unique physical blobs only for retirement conclusions. Early retirements overrepresent short-lived blobs; live unread data may still be recalled. Atime identifies any reader. No server restart/cleanup in this interval. Filesystem payload, not NVMe device writes. Different observer start times prevent matched-rank cohort comparisons.','ranks':reports};(p/'physical-recall-summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
