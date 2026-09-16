import importlib.util,json
from pathlib import Path
import torch
mods={}
for mode in ['original','fixed']:
 spec=importlib.util.spec_from_file_location('qsa_'+mode,'/e/qsa_ops.'+mode+'.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);mods[mode]=m
# Actual TP2 head dimensions. Identical query/cache/selection, different batched M.
torch.manual_seed(31)
q=torch.randn(1,24,128,device='cuda',dtype=torch.bfloat16)*0.3
k=torch.randn(2,1600,2,128,device='cuda',dtype=torch.bfloat16)*0.3
v=torch.randn_like(k)*0.2
indices=torch.cat([torch.arange(1023,device='cuda',dtype=torch.int32),torch.tensor([-1],device='cuda',dtype=torch.int32)])[None]
blocks=torch.tensor([[0,1]],device='cuda',dtype=torch.int32)
results=[]
for mode,mod in mods.items():
 ref=None
 for M in [1,4,330,991]:
  out=mod.qsa_sparse_paged_attention(q.repeat(M,1,1),k,v,indices.repeat(M,1),blocks,torch.zeros(M,device='cuda',dtype=torch.int32))
  if ref is None:ref=out[0].clone()
  d=(out.float()-ref.float()).abs();results.append({'method':mode,'M':M,'equal_to_C1':bool(torch.all(out==ref)),'changed':int((d!=0).sum()),'max_abs':float(d.max())})
  if mode=='fixed':assert torch.all(out==ref)
# FP32 mathematical reference for valid selected tokens (GQA repetition).
keys=k[0,:1023].repeat_interleave(12,dim=1).float();values=v[0,:1023].repeat_interleave(12,dim=1).float()
scores=torch.einsum('hd,thd->ht',q[0].float(),keys)/128**0.5
expected=torch.einsum('ht,thd->hd',scores.softmax(-1),values)
results.append({'test':'fixed versus FP32 reference','max_abs':float((ref.float()-expected).abs().max())})
assert torch.allclose(ref.float(),expected,atol=2e-4,rtol=.03)
Path('/e/qsa-attention-check.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
