import json,torch
from pathlib import Path
from safetensors import safe_open
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLEGroupedNorm
root=Path('/model-store/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47');idx=json.loads((root/'model.safetensors.index.json').read_text())['weight_map'];cfg=json.loads((root/'config.json').read_text())['text_config']
traces=[torch.load(p,weights_only=False) for p in Path('/e/traces-deep-r0').glob('*.pt')]
rows=[]
for t in traces:
 if 'serial-before' not in t['phase']:continue
 values={}
 for suffix in ['norm_key','norm_query','norm_conv']:
  name='language_model.model.layers.1.ple.'+suffix
  e=next(e for e in t['layers'] if e['name']==name and e['event']=='input');x=e['values']['args'][0].cuda()
  key='model.language_model.layers.1.ple.'+suffix+'.weight'
  with safe_open(root/idx[key],framework='pt',device='cpu') as f:w=f.get_tensor(key).cuda()
  layer=Qwen4ExpPLEGroupedNorm(len(w),cfg['rms_norm_eps'],cfg['hidden_size'],torch.bfloat16);layer.weight=torch.nn.Parameter(w,requires_grad=False)
  ref=layer(x);values[suffix]=ref
  for M in [4,330,991]:
   out=layer(x.repeat(M,1));d=(out.float()-ref.float()).abs();rows.append({'step':t['step'],'op':suffix,'M':M,'equal':bool(torch.all(out==ref)),'max_abs':float(d.max())})
 key=values['norm_key'].reshape(1,4,2560);query=values['norm_query'].reshape(1,4,2560)
 prod=key*query;ref=prod.sum(-1)
 for M in [4,330,991]:
  out=prod.repeat(M,1,1).sum(-1);rows.append({'step':t['step'],'op':'gate sum','M':M,'equal':bool(torch.all(out==ref)),'max_abs':float((out.float()-ref.float()).abs().max())})
Path('/e/ple-norm-check.json').write_text(json.dumps(rows,indent=2));print('checks',len(rows),'failures',[r for r in rows if not r['equal']])
