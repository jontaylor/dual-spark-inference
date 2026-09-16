import json,sys,torch
from pathlib import Path
root=Path(sys.argv[1]);traces=[torch.load(p,weights_only=False) for p in sorted(root.glob('*.pt'))]
refs={}
for t in traces:
 if 'serial-before' in t['phase']:
  pos=int(t['batch']['num_computed_tokens_np'][0])+int(t['batch']['num_scheduled_tokens'][0]);refs[pos]=t
rows=[]
for t in traces:
 if 'parallel' not in t['phase']:continue
 for i,rid in enumerate(t['batch']['req_ids']):
  pos=int(t['batch']['num_computed_tokens_np'][i])+int(t['batch']['num_scheduled_tokens'][i]);s=refs.get(pos)
  if s is None or int(t['batch']['num_scheduled_tokens'][i])!=int(s['batch']['num_scheduled_tokens'][0]):continue
  ref={(h['name'],h['event'],h['path']):h for h in s.get('row_hashes',[])};changes=[]
  for h in t.get('row_hashes',[]):
   key=(h['name'],h['event'],h['path']);a=ref.get(key)
   if a is None or a['shape']!=h['shape'] or h['event']!='output':continue
   if a['hashes'][0]!=h['hashes'][i]:changes.append({'name':h['name'],'event':h['event'],'path':h['path'],'shape':h['shape']})
  rows.append({'step':t['step'],'request':rid,'position':pos,'prefix_equal':torch.equal(s['state_token_ids'][0][:pos],t['state_token_ids'][i][:pos]),'first_differing_outputs':changes[:12]})
(root/'hash-analysis.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
