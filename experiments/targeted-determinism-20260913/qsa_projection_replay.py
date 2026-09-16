import json
from pathlib import Path
import torch
from safetensors import safe_open
from vllm.model_executor.determinism.gb10_targeted import fixed_bf16_matmul
root=Path('/model-store/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47')
cfg=json.loads((root/'config.json').read_text())['text_config'];index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
prefix='model.language_model.layers.3.self_attn.'
keys=[k for k in index if k.startswith(prefix)];print(keys,flush=True)
def weight(k):
 with safe_open(root/index[k],framework='pt',device='cpu') as f:return f.get_tensor(k)
t=torch.load('/e/traces-r0/374-0001.pt',weights_only=False)
x=next(e for e in t['layers'] if e['name']=='language_model.model.layers.3.self_attn' and e['event']=='input')['values']['kwargs']['hidden_states'].cuda()
parts=[]
for name in ('q_proj','k_proj','v_proj'):
 w=weight(prefix+name+'.weight');parts.append(w.chunk(2,dim=0)[0])
w=torch.cat(parts).cuda();print('qkv',w.shape,w.dtype,flush=True)
records=[]
for small,large in [(1,4),(330,991)]:
 a=x.repeat(small,1);b=x.repeat(large,1)
 for kind,fn in [('stock',lambda v:torch.nn.functional.linear(v,w)),('fixed',lambda v:fixed_bf16_matmul(v,w.t()))]:
  u=fn(a)[-1];v=fn(b)[-1];d=(u.float()-v.float()).abs()
  records.append({'projection':'layer3.qkv','input':'captured final prefill row repeated to isolate M','M':[small,large],'method':kind,'equal':torch.equal(u,v),'changed':int((d!=0).sum()),'max_abs':float(d.max())})
Path('/e/qsa-projection-replay.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
