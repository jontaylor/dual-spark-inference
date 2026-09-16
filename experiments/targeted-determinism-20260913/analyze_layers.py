"""Compare corresponding last-prefill-token activations by module boundary."""
import json
from pathlib import Path
import sys
import torch

root=Path(sys.argv[1])
traces=[torch.load(p,weights_only=False) for p in sorted(root.glob('*.pt'))]
serial=next(t for t in traces if 'serial-before-0' in t['batch']['req_ids'][0]
            and t.get('layers') and int(t['batch']['num_scheduled_tokens'][0])>1)
mixed=next(t for t in traces if t['batch']['num_reqs']==4 and t.get('layers')
           and t['batch']['num_scheduled_tokens'].tolist()==[1,330,330,330])

def tensors(x,path=''):
    if isinstance(x,torch.Tensor):return {path:x}
    if isinstance(x,dict):
        return {k:v for name,value in x.items() for k,v in tensors(value,path+'/'+str(name)).items()}
    if isinstance(x,list):
        return {k:v for i,value in enumerate(x) for k,v in tensors(value,path+'/'+str(i)).items()}
    return {}

base={(r['name'],r['event']):r for r in serial['layers']}
comparisons=[]
for event in mixed['layers']:
    key=(event['name'],event['event'])
    if key not in base:continue
    x=tensors(base[key]['values']); y=tensors(event['values'])
    for path,a in x.items():
        if path not in y:continue
        b=y[path]
        if not a.ndim or a.shape[0]!=1 or b.shape[0]!=4:continue
        for row in [1,2,3]:
            aa=a[0].float();bb=b[row].float()
            if aa.shape!=bb.shape:continue
            diff=bb-aa
            comparisons.append({'module':key[0],'event':key[1],'tensor':path,
                'row':row,'shape':list(aa.shape),'equal':torch.equal(aa,bb),
                'max_abs':float(diff.abs().max()),
                'relative_l2':float(diff.norm()/aa.norm().clamp_min(1e-20))})

out={'serial_step':serial['step'],'mixed_step':mixed['step'],
     'serial_hook_events':len(serial['layers']),'mixed_hook_events':len(mixed['layers']),
     'hook_errors':[e for t in traces for e in t.get('layer_errors',[])],
     'comparisons':comparisons}
(root/'layer-analysis.json').write_text(json.dumps(out,indent=2))
print({k:v for k,v in out.items() if k!='comparisons'})
for row in [1,2,3]:
    changed=[r for r in comparisons if r['row']==row and not r['equal']]
    print('ROW',row,'FIRST CHANGES',json.dumps(changed[:12]))
