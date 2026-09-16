import json,torch
from pathlib import Path
from vllm.model_executor.determinism.gb10_targeted import fixed_bf16_matmul
s=m=None
for path in sorted(Path('/e/traces-scalar-r0').glob('*.pt')):
 t=torch.load(path,weights_only=False)
 if t['phase']=='captured-scalar-serial-before' and int(t['batch']['num_scheduled_tokens'][0])==330:s=t
 if t['phase']=='captured-scalar-parallel' and t['batch']['num_reqs']==4 and t['batch']['has_prefill']:m=t
 del t
assert s is not None and m is not None
name='language_model.model.layers.44.mlp.shared_expert_gate'
def x(t):return next(e for e in t['full_ops'] if e['name']==name and e['event']=='input')['values']['args'][0].cuda()
a=x(s);b=x(m);w=torch.load('/e/weights-scalar-r0/'+name+'.pt',weights_only=False)['weight'].cuda()
assert torch.equal(a,b[1:331]);records=[]
for kind,fn in [('stock',lambda x:torch.nn.functional.linear(x,w)),('fixed',lambda x:fixed_bf16_matmul(x,w.t()))]:
 u=fn(a);v=fn(b)[1:331];d=(u.float()-v.float()).abs();changed=torch.nonzero(d).cpu().tolist()
 records.append({'method':kind,'M':[len(a),len(b)],'inputs_equal':True,'equal':torch.equal(u,v),'max_abs':float(d.max()),'changed_indices':changed})
 if kind=='fixed':assert torch.equal(u,v)
Path('/e/scalar-projection-check.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
