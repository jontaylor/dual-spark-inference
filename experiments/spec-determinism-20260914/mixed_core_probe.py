import json,types,torch,sys,importlib.util
from unittest.mock import patch
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as m
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices,prepare_chunk_offsets
spec=importlib.util.spec_from_file_location('spec_fixed','/tmp/fused_sigmoid.fixed.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
m.fused_sigmoid_gating_delta_rule_update=mod.fused_sigmoid_gating_delta_rule_update
H,HV,K,V=8,24,128,128;D=2*H*K+HV*V;L=330;T=4;dev='cuda';dt=torch.bfloat16;prefix='test'
torch.manual_seed(27);x=torch.randn(L,D,device=dev,dtype=dt);a=torch.randn(L,HV,device=dev,dtype=dt);b=torch.randn_like(a);cw=torch.randn(D,1,4,device=dev,dtype=dt)*.1;A=torch.randn(HV,device=dev);bias=torch.randn_like(A)
methods={n:getattr(m.QwenGatedDeltaNetAttention,n) for n in ['rearrange_mixed_qkv','_forward_core','_forward_core_decode_non_spec','_forward_packed_recurrent_decode']};conv_fn=m.causal_conv1d_fn
records=[]
for sd in [torch.float32,torch.bfloat16]:
 cs=torch.randn(1,D,6,device=dev,dtype=dt);ss=torch.randn(1,HV,V,K,device=dev,dtype=sd)*.1
 def run(n,p):
  total=n*T+p*L;slots=1+n*T+p;ids=torch.arange(1,n*T+1,device=dev,dtype=torch.int32).reshape(n,T);pi=torch.arange(n*T+1,slots,device=dev,dtype=torch.int32);pcu=torch.arange(p+1,dtype=torch.int32)*L
  meta=GDNAttentionMetadata(num_prefills=p,num_prefill_tokens=p*L,num_decodes=0,num_decode_tokens=0,num_spec_decodes=n,num_spec_decode_tokens=n*T,num_actual_tokens=total,spec_query_start_loc=torch.arange(n+1,device=dev,dtype=torch.int32)*T,spec_sequence_masks=torch.tensor([True]*n+[False]*p,device=dev),spec_token_indx=torch.arange(n*T,device=dev),non_spec_token_indx=torch.arange(n*T,total,device=dev),spec_state_indices_tensor=ids,num_accepted_tokens=torch.ones(n,device=dev,dtype=torch.int32),non_spec_query_start_loc=pcu.cuda(),non_spec_state_indices_tensor=pi,has_initial_state=torch.ones(p,device=dev,dtype=torch.bool),prefill_query_start_loc=pcu.cuda(),prefill_state_indices=pi,prefill_has_initial_state=torch.ones(p,device=dev,dtype=torch.bool),chunk_indices=prepare_chunk_indices(pcu,64).cuda() if p else None,chunk_offsets=prepare_chunk_offsets(pcu,64).cuda() if p else None)
  conv=cs.repeat(slots,1,1)
  if not m.is_conv_state_dim_first():conv=conv.transpose(-1,-2).contiguous()
  layer=types.SimpleNamespace(prefix=prefix,enable_packed_recurrent_decode=True,tp_size=1,num_k_heads=H,num_v_heads=HV,head_k_dim=K,head_v_dim=V,key_dim=H*K,value_dim=HV*V,activation='silu',A_log=A,dt_bias=bias,conv1d=types.SimpleNamespace(weight=cw,bias=None),kv_cache=(conv,ss.repeat(slots,1,1,1)),chunk_gated_delta_rule=m.fla_chunk_gated_delta_rule)
  for name,fn in methods.items():setattr(layer,name,types.MethodType(fn,layer))
  out=torch.zeros(total,HV,V,device=dev,dtype=dt)
  with patch.object(m,'get_forward_context',return_value=types.SimpleNamespace(attn_metadata={prefix:meta})),patch.object(m,'causal_conv1d_fn',side_effect=lambda *args,**kw:conv_fn(*args,**{k:v for k,v in kw.items() if k!='metadata'})):
   layer._forward_core(torch.cat([x[:T].repeat(n,1),x.repeat(p,1)]),torch.cat([b[:T].repeat(n,1),b.repeat(p,1)]),torch.cat([a[:T].repeat(n,1),a.repeat(p,1)]),out)
  return out,layer.kv_cache[1]
 u,us=run(1,0)
 for n,p in [(4,0),(1,3),(4,2)]:
  v,vs=run(n,p)
  records.append(dict(dtype=str(sd),spec_sequences=n,prefills=p,output_equal=all(torch.equal(u,v[j*T:(j+1)*T]) for j in range(n)),state_equal=all(torch.equal(us[1:],vs[1+j*T:1+(j+1)*T]) for j in range(n))))
print(json.dumps(records,indent=2));assert all(r['output_equal'] and r['state_equal'] for r in records)
