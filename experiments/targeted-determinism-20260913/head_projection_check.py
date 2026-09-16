import json,torch
from pathlib import Path
from safetensors import safe_open
from vllm.model_executor.determinism.gb10_targeted import fixed_bf16_matmul
from vllm.model_executor.determinism.batch_invariant import matmul_kernel_persistent
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units
root=Path('/model-store/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47');idx=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
keys=[k for k in idx if 'lm_head' in k];print(keys,flush=True)
key=keys[0]
with safe_open(root/idx[key],framework='pt',device='cpu') as f:
 full=f.get_tensor(key);w=full.chunk(2,dim=0)[0].cuda()
t=next(t for p in Path('/e/traces-deep-r0').glob('*.pt') if (t:=torch.load(p,weights_only=False))['phase']=='captured-deep-serial-before' and int(t['batch']['num_computed_tokens_np'][0])==1600)
x=t['sample_hidden_states'].cuda();print(x.shape,x.dtype,w.shape,w.dtype,flush=True)
def small(a):
 b=w.t();M,K=a.shape;N=b.shape[1];c=torch.empty(M,N,device=a.device,dtype=a.dtype);sms=num_compute_units(a.device.index)
 matmul_kernel_persistent[(min(sms,triton.cdiv(M,16)*triton.cdiv(N,128)),)](a,b,c,None,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),c.stride(0),c.stride(1),NUM_SMS=sms,A_LARGE=False,B_LARGE=False,C_LARGE=False,HAS_BIAS=False,BLOCK_SIZE_M=16,BLOCK_SIZE_N=128,BLOCK_SIZE_K=64,GROUP_SIZE_M=8,num_stages=3,num_warps=4)
 return c
records=[];target=fixed_bf16_matmul(x,w.t())
for name,fn in [('stock',lambda a:torch.nn.functional.linear(a,w)),('fixed128',lambda a:fixed_bf16_matmul(a,w.t())),('fixed16',small)]:
 a=fn(x);b=fn(x.repeat(4,1));d=(b.float()-a.float()).abs()
 records.append({'method':name,'C1_C4_equal':bool(torch.all(b==a)),'changed':int((d!=0).sum()),'max_abs':float(d.max()),'matches_fixed128':torch.equal(a,target)})
 if name!='stock':assert torch.all(b==a)
 if name=='stock':records.append({'stock_C1_matches_captured_rank0_logits':torch.equal(a[0].float().cpu(),t['raw_logits'][0,:w.shape[0]].float()), 'captured_max_abs':float((a[0].float().cpu()-t['raw_logits'][0,:w.shape[0]].float()).abs().max()), 'captured_changed':int((a[0].float().cpu()!=t['raw_logits'][0,:w.shape[0]].float()).sum())})
Path('/e/head-projection-check.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
