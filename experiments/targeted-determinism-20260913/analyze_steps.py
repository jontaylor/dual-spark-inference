import json,sys,torch
from pathlib import Path
root=Path(sys.argv[1]);traces=[torch.load(p,weights_only=False) for p in sorted(root.glob('*.pt'))]
def tensors(x,p=''):
 if isinstance(x,torch.Tensor):return {p:x}
 if isinstance(x,dict):return {k:v for a,b in x.items() for k,v in tensors(b,p+'/'+str(a)).items()}
 if isinstance(x,(list,tuple)):return {k:v for a,b in enumerate(x) for k,v in tensors(b,p+'/'+str(a)).items()}
 return {}
refs={}
for t in traces:
 if 'serial-before' in t['phase'] and t.get('raw_logits') is not None:
  computed=int(t['batch']['num_computed_tokens_np'][0]);scheduled=int(t['batch']['num_scheduled_tokens'][0]);refs[computed+scheduled]=t
rows=[]
for t in traces:
 if 'parallel' not in t['phase'] or t.get('raw_logits') is None:continue
 for i,rid in enumerate(t['batch']['req_ids']):
  computed=int(t['batch']['num_computed_tokens_np'][i]);scheduled=int(t['batch']['num_scheduled_tokens'][i]);pos=computed+scheduled
  if pos not in refs:continue
  s=refs[pos];prefix_equal=torch.equal(s['state_token_ids'][0][:pos],t['state_token_ids'][i][:pos])
  base={(e['name'],e['event']):e for e in s['layers']};changes=[]
  for e in t['layers']:
   key=(e['name'],e['event']);r=base.get(key)
   if r is None:continue
   a=tensors(r['values']);b=tensors(e['values'])
   for name,x in a.items():
    if key[0].endswith('.indexer') and key[1]=='input' and name=='/args/2':continue
    y=b.get(name)
    if y is None or not x.ndim or x.shape[0]!=1 or y.shape[0]!=t['batch']['num_reqs']:continue
    if x[0].shape!=y[i].shape:continue
    if not torch.equal(x[0],y[i]):
     d=(x[0].float()-y[i].float()).abs();changes.append({'name':key[0],'event':key[1],'tensor':name,'changed':int((d!=0).sum()),'max_abs':float(d.max())})
  row={'step':t['step'],'request':rid,'position':pos,'prefix_equal':prefix_equal,'first_changes':changes[:5]}
  for name in ('sample_hidden_states','raw_logits'):
   if name in s and name in t:
    a=s[name][0];b=t[name][i];row[name+'_equal']=torch.equal(a,b);row[name+'_max_abs']=float((a.float()-b.float()).abs().max())
  rows.append(row)
(root/'step-analysis.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
