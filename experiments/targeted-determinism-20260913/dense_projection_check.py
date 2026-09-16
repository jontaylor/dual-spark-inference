import json,torch
from pathlib import Path
from safetensors import safe_open
from vllm.model_executor.determinism.gb10_targeted import fixed_bf16_matmul
traces=[torch.load(p,weights_only=False) for p in Path('/e/traces-dense-r0').glob('*.pt')]
t=next(t for t in traces if t['phase']=='captured-dense-serial-before' and int(t['batch']['num_computed_tokens_np'][0])==1930)
root=Path('/model-store/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47');idx=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
records=[]
for suffix in ['0.mlp.gate','0.mlp.shared_expert.gate_up_proj','0.mlp.shared_expert.down_proj','0.mlp.shared_expert_gate','1.ple.key_proj','1.ple.value_proj']:
 name='language_model.model.layers.'+suffix
 e=next(e for e in t['layers'] if e['name']==name and e['event']=='input')
 x=e['values']['args'][0].cuda()
 if suffix.startswith('0.'):
  wdata=torch.load('/e/weights-dense-r0/'+name+'.pt',weights_only=False);w=wdata['weight'].cuda();bias=wdata.get('bias');bias=bias.cuda() if bias is not None else None
 else:
  key='model.language_model.layers.'+suffix+'.weight'
  with safe_open(root/idx[key],framework='pt',device='cpu') as f:w=f.get_tensor(key).cuda()
  bias=None
 print(name,x.shape,x.dtype,w.shape,w.dtype,flush=True)
 for small,large in [(1,4),(1,991),(330,991)]:
  for kind,fn in [('stock',lambda a:torch.nn.functional.linear(a,w,bias)),('fixed',lambda a:fixed_bf16_matmul(a,w.t(),bias))]:
   u=fn(x.repeat(small,1))[-1];v=fn(x.repeat(large,1))[-1];d=(u.float()-v.float()).abs()
   records.append({'name':name,'M':[small,large],'method':kind,'equal':torch.equal(u,v),'changed':int((d!=0).sum()),'max_abs':float(d.max())})
   if kind=='fixed':assert torch.equal(u,v)
Path('/e/dense-projection-check.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
