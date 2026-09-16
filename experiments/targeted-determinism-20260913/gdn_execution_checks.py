"""Regression boundaries from PR #49827, exercised on the GB10 port."""
import ast,json,sys,types
from pathlib import Path
from unittest.mock import patch
import torch
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as qmod
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices,prepare_chunk_offsets

MODE=sys.argv[1]
methods={n:getattr(qmod.QwenGatedDeltaNetAttention,n) for n in ['rearrange_mixed_qkv','_forward_core','_forward_core_decode_non_spec','_forward_packed_recurrent_decode']}
if MODE=='baseline':
    tree=ast.parse(Path('/e/gdn.original.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QwenGatedDeltaNetAttention')
    nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in methods]
    ns=vars(qmod);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<baseline methods>','exec'),ns)
    methods.update({n.name:ns[n.name] for n in nodes})
torch.manual_seed(11);device='cuda';dtype=torch.bfloat16
H,HV,K,V=4,8,128,128;dim=2*H*K+HV*V;prefix='test.linear_attn'
records=[]
for state_dtype in [torch.float32,torch.bfloat16]:
    state=torch.randn(3,HV,K,V,device=device,dtype=state_dtype)*0.05
    conv=torch.zeros(3,dim,3,device=device,dtype=dtype)
    A=torch.randn(HV,device=device)*0.1;dt=torch.randn(HV,device=device)*0.1
    x=torch.randn(5,dim,device=device,dtype=dtype)*0.1
    a=torch.randn(5,HV,device=device,dtype=dtype)*0.1;b=torch.randn_like(a)*0.1
    ids=torch.arange(3,device=device,dtype=torch.int32)
    decode=GDNAttentionMetadata(num_prefills=0,num_prefill_tokens=0,num_decodes=2,num_decode_tokens=2,num_spec_decodes=0,num_spec_decode_tokens=0,num_actual_tokens=2,non_spec_query_start_loc=torch.arange(3,device=device,dtype=torch.int32),non_spec_state_indices_tensor=ids[:2])
    mixed=GDNAttentionMetadata(num_prefills=1,num_prefill_tokens=3,num_decodes=2,num_decode_tokens=2,num_spec_decodes=0,num_spec_decode_tokens=0,num_actual_tokens=5,has_initial_state=torch.tensor([True,True,False],device=device),non_spec_query_start_loc=torch.tensor([0,1,2,5],device=device,dtype=torch.int32),non_spec_state_indices_tensor=ids,prefill_query_start_loc=torch.tensor([0,3],device=device,dtype=torch.int32),prefill_state_indices=ids[2:],prefill_has_initial_state=torch.tensor([False],device=device))
    def run(meta):
        layer=types.SimpleNamespace(prefix=prefix,enable_packed_recurrent_decode=True,tp_size=1,num_k_heads=H,num_v_heads=HV,head_k_dim=K,head_v_dim=V,key_dim=H*K,value_dim=HV*V,activation='silu',A_log=A,dt_bias=dt,conv1d=types.SimpleNamespace(weight=torch.zeros(dim,1,4,device=device,dtype=dtype),bias=None),kv_cache=(conv.clone(),state.clone()))
        layer.chunk_gated_delta_rule=lambda **kw:(torch.zeros_like(kw['v']),kw['initial_state'].clone())
        for name,fn in methods.items():setattr(layer,name,types.MethodType(fn,layer))
        out=torch.zeros(meta.num_actual_tokens,HV,V,device=device,dtype=dtype)
        with patch.object(qmod,'get_forward_context',return_value=types.SimpleNamespace(attn_metadata={prefix:meta})),patch.object(qmod,'causal_conv1d_fn',side_effect=lambda x,*args,**kw:x),patch.object(qmod,'causal_conv1d_update',side_effect=lambda x,*args,**kw:x):
            layer._forward_core(x.clone(),b.clone(),a.clone(),out)
        return out,layer.kv_cache[1]
    u,us=run(decode);v,vs=run(mixed)
    records.append({'test':'packed recurrent mixed route','state_dtype':str(state_dtype),'output_equal':torch.equal(u,v[:2]),'state_equal':torch.equal(us[:2],vs[:2]),'scope':'real recurrent kernels; convolution and fresh-prefill work stubbed to isolate routing'})

q=torch.randn(1,192,HV,K,device=device,dtype=dtype)*0.1;k=torch.randn_like(q)*0.1;v=torch.randn_like(q)*0.1
g=-torch.rand(1,192,HV,device=device)*0.1;beta=torch.rand(1,192,HV,device=device,dtype=dtype);state=torch.randn(1,HV,K,V,device=device)*0.01

def run_chunks(ends):
    outputs=[];s=state.clone();start=0
    for end in ends:
        cu=torch.tensor([0,end-start],dtype=torch.int32)
        out,s=qmod.fla_chunk_gated_delta_rule(q=q[:,start:end].contiguous(),k=k[:,start:end].contiguous(),v=v[:,start:end].contiguous(),g=g[:,start:end].contiguous(),beta=beta[:,start:end].contiguous(),initial_state=s.contiguous(),output_final_state=True,cu_seqlens=cu.cuda(),chunk_indices=prepare_chunk_indices(cu,64).cuda(),chunk_offsets=prepare_chunk_offsets(cu,64).cuda(),use_qk_l2norm_in_kernel=False)
        outputs.append(out);start=end
    return torch.cat(outputs,dim=1),s
full,fs=run_chunks([192])
for ends in [[64,128,192],[73,128,192]]:
    out,s=run_chunks(ends);records.append({'test':'Triton prefill partition','ends':ends,'output_equal':torch.equal(full,out),'state_equal':torch.equal(fs,s)})

def batched(n):
    cu=torch.arange(n+1,dtype=torch.int32)*64
    return qmod.fla_chunk_gated_delta_rule(q=q[:,:64].repeat(1,n,1,1).contiguous(),k=k[:,:64].repeat(1,n,1,1).contiguous(),v=v[:,:64].repeat(1,n,1,1).contiguous(),g=g[:,:64].repeat(1,n,1).contiguous(),beta=beta[:,:64].repeat(1,n,1).contiguous(),initial_state=state.repeat(n,1,1,1).contiguous(),output_final_state=True,cu_seqlens=cu.cuda(),chunk_indices=prepare_chunk_indices(cu,64).cuda(),chunk_offsets=prepare_chunk_offsets(cu,64).cuda(),use_qk_l2norm_in_kernel=False)
u,us=batched(1);v,vs=batched(4);records.append({'test':'Triton prefill C1 C4','output_equal':all(torch.equal(u,v[:,i*64:(i+1)*64]) for i in range(4)),'state_equal':all(torch.equal(us[0],vs[i]) for i in range(4))})
# FI uses a hardware-specific implementation; verify the port pins its selector.
seen=[]
def fake_fi(**kw):seen.append(kw['use_cp']);return kw['v'],kw['initial_state']
with patch('flashinfer.gdn_prefill.chunk_gated_delta_rule',side_effect=fake_fi):
    qmod.fi_chunk_gated_delta_rule(q=q[:,:64],k=k[:,:64],v=q[:,:64],g=g[:,:64],beta=beta[:,:64],initial_state=state,output_final_state=True,cu_seqlens=torch.tensor([0,64],device=device,dtype=torch.int32),use_qk_l2norm_in_kernel=False)
records.append({'test':'FlashInfer context-parallel selection','use_cp':seen,'scope':'argument-level check; production GB10 prefill uses Triton'})
Path('/e/gdn-execution-'+MODE+'.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
if MODE=='patched':
    for rec in records:
        if rec.get('ends')==[73,128,192]:assert not(rec['output_equal'] and rec['state_equal'])
        elif 'output_equal' in rec:assert rec['output_equal'] and rec['state_equal'],rec
    assert seen==[False]
