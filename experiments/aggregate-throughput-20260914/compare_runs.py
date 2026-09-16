import json,statistics
from pathlib import Path
root=Path('/home/jon/artifact-agent-orchestration/.artifact-agent/baseline-comparison');out=[]
for name in ['20260913T182147Z-temperature','20260914T004328Z-temperature-recovery-repeat']:
 p=root/name;rows=[]
 for f in p.glob('temperature-*/model-output-*.json'):
  d=json.loads(f.read_text());u=d.get('usage') or {};created=d.get('created',0)
  if not u:continue
  # Isolate current MTP epoch, excluding previous no-spec and startup windows.
  if name.startswith('20260914') and created<1789348400:continue
  t=f.with_name(f.name.replace('model-output-','model-timing-'));tim=json.loads(t.read_text()) if t.exists() else {}
  rows.append({'trial':f.parent.name,'created':created,'prompt':u.get('prompt_tokens',0),'generated':u.get('completion_tokens',0),'cached':(u.get('prompt_tokens_details') or {}).get('cached_tokens',0),'seconds':tim.get('elapsed_seconds')})
 total=lambda k:sum(r[k] for r in rows)
 out.append({'run':name,'requests':len(rows),'prompt_tokens':total('prompt'),'cached_tokens':total('cached'),'cache_fraction':total('cached')/max(1,total('prompt')),'computed_prompt_tokens':total('prompt')-total('cached'),'generated_tokens':total('generated'),'median_request_seconds':statistics.median(r['seconds'] for r in rows if r['seconds'] is not None) if rows else None,'rows':rows})
print(json.dumps(out))
