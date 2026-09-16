import subprocess,json
from pathlib import Path
p=Path(__file__).resolve().parent
code='''import json,time
from pathlib import Path
p=Path('/home/jon/artifact-agent-orchestration/.artifact-agent/baseline-comparison/20260914T004328Z-temperature-recovery-repeat');start=json.loads((p/'aligned-cache-validation-resume.json').read_text())['time'];rows=[]
for f in p.glob('temperature-*/model-output-*.json'):
 try:d=json.loads(f.read_text())
 except (ValueError,OSError):continue
 if d.get('created',0)<start:continue
 u=d.get('usage') or {}
 if not u:continue
 rows.append({'file':str(f.relative_to(p)),'created':d.get('created'),'prompt':u.get('prompt_tokens',0),'cached':(u.get('prompt_tokens_details') or {}).get('cached_tokens',0),'generated':u.get('completion_tokens',0)})
print(json.dumps({'time':time.time(),'resume_time':start,'rows':sorted(rows,key=lambda r:r['created'])},indent=2))
'''
r=subprocess.run(['ssh','jon@192.168.0.167','python3 -'],input=code,text=True,capture_output=True,check=True,timeout=30)
d=json.loads(r.stdout);(p/'post-resume-responses.json').write_text(r.stdout)
# First request per arm rebuilds cache after restart. Subsequent completed
# requests form a continuation subset, not a replacement for whole-interval stats.
seen=set();continued=[];first=[]
for row in d['rows']:
 arm=row['file'].split('/')[0]
 (first if arm not in seen else continued).append(row);seen.add(arm)
def summarize(rows):
 prompt=sum(r['prompt'] for r in rows);cached=sum(r['cached'] for r in rows)
 return {'requests':len(rows),'prompt':prompt,'cached':cached,'computed':prompt-cached,'cache_fraction':cached/prompt if prompt else None,'generated':sum(r['generated'] for r in rows)}
s={'all_completed_since_resume':summarize(d['rows']),'first_completed_per_arm':summarize(first),'subsequent_completed_per_arm':summarize(continued)}
(p/'post-resume-response-summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
