import json
from pathlib import Path
import sys
import torch

root=Path(sys.argv[1])
rs=[torch.load(p,weights_only=False) for p in sorted(root.glob('*.pt'))]
s=next(t for t in rs if t.get('full_ops') and 'serial-before-0-' in t['batch']['req_ids'][0])
m=next(t for t in rs if t.get('full_ops') and t['batch']['num_reqs']==4 and t['batch']['has_prefill'])
def flat(x,path=''):
    if isinstance(x,torch.Tensor):return {path:x}
    if isinstance(x,dict):return {k:v for n,w in x.items() for k,v in flat(w,path+'/'+str(n)).items()}
    if isinstance(x,list):return {k:v for n,w in enumerate(x) for k,v in flat(w,path+'/'+str(n)).items()}
    return {}
base={(e['name'],e['event']):e for e in s['full_ops']}
out=[];states=[]
for e in m['full_ops']:
    key=(e['name'],e['event'])
    if key not in base:continue
    x=flat(base[key]['values']);y=flat(e['values'])
    for i in range(m['batch']['num_reqs']):
        a,z=map(int,m['batch']['query_start_loc_np'][i:i+2])
        if z-a!=s['batch']['num_tokens']:continue
        if 'initial_cache' in e:
            for j,(aa,bb) in enumerate(zip(base[key]['initial_cache'],e['initial_cache'])):
                d=bb[i].float()-aa[0].float()
                states.append({'module':key[0],'row':i,'state':j,'equal':torch.equal(aa[0],bb[i]),'max_abs':float(d.abs().max()),'relative_l2':float(d.norm()/aa[0].float().norm().clamp_min(1e-20))})
        for path,aa in x.items():
            if path not in y:continue
            bb=y[path]
            if not aa.ndim or aa.shape[0]%s['batch']['num_tokens']:continue
            factor=aa.shape[0]//s['batch']['num_tokens']
            if bb.shape[0]!=m['batch']['num_tokens']*factor:continue
            bb=bb[a*factor:z*factor]
            if aa.shape!=bb.shape:continue
            d=bb.float()-aa.float()
            changed=(d!=0).reshape(len(aa),-1).any(dim=1).nonzero().flatten()
            out.append({'module':key[0],'event':key[1],'tensor':path,'row':i,
                'equal':torch.equal(aa,bb),'max_abs':float(d.abs().max()),
                'relative_l2':float(d.norm()/aa.float().norm().clamp_min(1e-20)),
                'changed_rows':len(changed),'first_changed_row':int(changed[0]) if len(changed) else None})
summary={'serial_step':s['step'],'mixed_step':m['step'],'hook_errors':[e for t in rs for e in t.get('layer_errors',[])],'states':states,'comparisons':out}
(root/'full-analysis.json').write_text(json.dumps(summary,indent=2))
print('STATE COMPARISONS',json.dumps(states,indent=2))
print('ROW 1 OPERATIONS',json.dumps([r for r in out if r['row']==1],indent=2))
