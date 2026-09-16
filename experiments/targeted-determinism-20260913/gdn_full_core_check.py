import json,types,torch
from pathlib import Path
from unittest.mock import patch
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as m
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices,prepare_chunk_offsets
H,HV,K,V=8,24,128,128;dim=2*H*K+HV*V;prefix='test';L=330
methods={n:getattr(m.QwenGatedDeltaNetAttention,n) for n in ['rearrange_mixed_qkv','_forward_core','_forward_core_decode_non_spec','_forward_packed_recurrent_decode']}
torch.manual_seed(43);dev='cuda';dtype=torch.bfloat16
x=torch.randn(L,dim,device=dev,dtype=dtype);a=torch.randn(L,HV,device=dev,dtype=dtype);b=torch.randn_like(a)
conv=torch.randn(1,dim,3,device=dev,dtype=dtype);state=torch.randn(1,HV,K,V,device=dev,dtype=torch.float32)*0.1
if not m.is_conv_state_dim_first():conv=conv.transpose(-1,-2).contiguous()
cw=torch.randn(dim,1,4,device=dev,dtype=dtype)*.1;A=torch.randn(HV,device=dev);dt=torch.randn(HV,device=dev)
conv_fn=m.causal_conv1d_fn

def run(n,mixed):
 nd=int(mixed);N=n+nd;cu=torch.arange(n+1,dtype=torch.int32)*L
 total=nd+n*L;noncu=torch.cat([torch.tensor([0],dtype=torch.int32),cu+nd]) if nd else cu
 ids=torch.arange(1,N+1,device=dev,dtype=torch.int32)
 meta=GDNAttentionMetadata(num_prefills=n,num_prefill_tokens=n*L,num_decodes=nd,num_decode_tokens=nd,num_spec_decodes=0,num_spec_decode_tokens=0,num_actual_tokens=total,has_initial_state=torch.ones(N,device=dev,dtype=torch.bool),non_spec_query_start_loc=noncu.cuda(),non_spec_state_indices_tensor=ids,prefill_query_start_loc=cu.cuda(),prefill_state_indices=ids[nd:],prefill_has_initial_state=torch.ones(n,device=dev,dtype=torch.bool),chunk_indices=prepare_chunk_indices(cu,64).cuda(),chunk_offsets=prepare_chunk_offsets(cu,64).cuda())
 layer=types.SimpleNamespace(prefix=prefix,enable_packed_recurrent_decode=True,tp_size=1,num_k_heads=H,num_v_heads=HV,head_k_dim=K,head_v_dim=V,key_dim=H*K,value_dim=HV*V,activation='silu',A_log=A,dt_bias=dt,conv1d=types.SimpleNamespace(weight=cw,bias=None),kv_cache=(conv.repeat(N+1,1,1),state.repeat(N+1,1,1,1)),chunk_gated_delta_rule=m.fla_chunk_gated_delta_rule)
 for name,fn in methods.items():setattr(layer,name,types.MethodType(fn,layer))
 out=torch.zeros(total,HV,V,device=dev,dtype=dtype)
 xx=x.repeat(n,1);aa=a.repeat(n,1);bb=b.repeat(n,1)
 if nd:xx=torch.cat([x[:1],xx]);aa=torch.cat([a[:1],aa]);bb=torch.cat([b[:1],bb])
 with patch.object(m,'get_forward_context',return_value=types.SimpleNamespace(attn_metadata={prefix:meta})),patch.object(m,'causal_conv1d_fn',side_effect=lambda *args,**kw:conv_fn(*args,**{k:v for k,v in kw.items() if k!='metadata'})):
  layer._forward_core(xx,bb,aa,out)
 return out[nd:],layer.kv_cache[1][nd+1:]
u,us=run(1,False);rows=[]
for n,mixed in [(4,False),(3,True)]:
 v,vs=run(n,mixed)
 for i in range(n):
  d=(u.float()-v[i*L:(i+1)*L].float()).abs();rows.append({'n_prefills':n,'mixed_decode':mixed,'row':i,'output_equal':torch.equal(u,v[i*L:(i+1)*L]),'max_abs':float(d.max()),'state_equal':torch.equal(us[0],vs[i])})
Path('/e/gdn-full-core-check.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))

assert all(r["output_equal"] and r["state_equal"] for r in rows)
