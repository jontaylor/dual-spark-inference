"""Regression boundaries from PR #49827, exercised on the GB10 port."""
import ast,json,sys,types
from pathlib import Path
from unittest.mock import patch
import torch
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qmod
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices,prepare_chunk_offsets

MODE='patched'
methods={n:getattr(qmod.QwenGatedDeltaNetAttention,n) for n in ['rearrange_mixed_qkv','_forward_core','_forward_core_decode_non_spec','_forward_packed_recurrent_decode']}
if MODE=='baseline':
    tree=ast.parse(Path('/e/gdn.original.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QwenGatedDeltaNetAttention')
    nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in methods]
    ns=vars(qmod);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<baseline methods>','exec'),ns)
    methods.update({n.name:ns[n.name] for n in nodes})
torch.manual_seed(11);device='cuda';dtype=torch.bfloat16
H,HV,K,V=8,24,128,128;dim=2*H*K+HV*V;prefix='test.linear_attn'
records=[]
conv_fn=qmod.causal_conv1d_fn
for state_dtype in [torch.float32,torch.bfloat16]:
    state=torch.randn(4,HV,K,V,device=device,dtype=state_dtype)*0.05
    conv=torch.randn(4,dim,3,device=device,dtype=dtype)*0.1
    if not qmod.is_conv_state_dim_first():conv=conv.transpose(-1,-2).contiguous()
    cw=torch.randn(dim,1,4,device=device,dtype=dtype)*0.1
    A=torch.randn(HV,device=device)*0.1;dt=torch.randn(HV,device=device)*0.1
    x=torch.randn(5,dim,device=device,dtype=dtype)*0.1
    a=torch.randn(5,HV,device=device,dtype=dtype)*0.1;b=torch.randn_like(a)*0.1
    ids=torch.arange(1,4,device=device,dtype=torch.int32)
    decode=GDNAttentionMetadata(num_prefills=0,num_prefill_tokens=0,num_decodes=2,num_decode_tokens=2,num_spec_decodes=0,num_spec_decode_tokens=0,num_actual_tokens=2,non_spec_query_start_loc=torch.arange(3,device=device,dtype=torch.int32),non_spec_state_indices_tensor=ids[:2])
    mixed=GDNAttentionMetadata(num_prefills=1,num_prefill_tokens=3,num_decodes=2,num_decode_tokens=2,num_spec_decodes=0,num_spec_decode_tokens=0,num_actual_tokens=5,has_initial_state=torch.tensor([True,True,False],device=device),non_spec_query_start_loc=torch.tensor([0,1,2,5],device=device,dtype=torch.int32),non_spec_state_indices_tensor=ids,prefill_query_start_loc=torch.tensor([0,3],device=device,dtype=torch.int32),prefill_state_indices=ids[2:],prefill_has_initial_state=torch.tensor([False],device=device))
    def run(meta):
        layer=types.SimpleNamespace(prefix=prefix,enable_packed_recurrent_decode=True,tp_size=1,num_k_heads=H,num_v_heads=HV,head_k_dim=K,head_v_dim=V,key_dim=H*K,value_dim=HV*V,activation='silu',A_log=A,dt_bias=dt,conv1d=types.SimpleNamespace(weight=cw,bias=None),kv_cache=(conv.clone(),state.clone()))
        layer.chunk_gated_delta_rule=lambda **kw:(torch.zeros_like(kw['v']),kw['initial_state'].clone())
        for name,fn in methods.items():setattr(layer,name,types.MethodType(fn,layer))
        out=torch.zeros(meta.num_actual_tokens,HV,V,device=device,dtype=dtype)
        with patch.object(qmod,'get_forward_context',return_value=types.SimpleNamespace(attn_metadata={prefix:meta})),patch.object(qmod,'causal_conv1d_fn',side_effect=lambda *args,**kw:conv_fn(*args,**{k:v for k,v in kw.items() if k!='metadata'})):
            layer._forward_core(x.clone(),b.clone(),a.clone(),out)
        return out,layer.kv_cache[1]
    u,us=run(decode);v,vs=run(mixed)
    records.append({'test':'packed recurrent mixed route','state_dtype':str(state_dtype),'output_equal':torch.equal(u,v[:2]),'state_equal':torch.equal(us[1:3],vs[1:3]),'scope':'real convolution and recurrent kernels; only new-prefill recurrence stubbed'})

Path('/e/gdn-real-conv-check.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))

assert all(r["output_equal"] and r["state_equal"] for r in records)
