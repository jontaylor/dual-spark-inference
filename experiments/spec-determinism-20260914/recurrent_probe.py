import json,torch,sys
if len(sys.argv)>1:
 import importlib.util
 spec=importlib.util.spec_from_file_location("spec_fixed", "/tmp/fused_sigmoid.fixed.py")
 mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
from vllm.third_party.flash_linear_attention.ops import fused_sigmoid_gating_delta_rule_update as f
if len(sys.argv)>1:f=mod.fused_sigmoid_gating_delta_rule_update
from vllm.third_party.flash_linear_attention.ops import fused_recurrent_gated_delta_rule_packed_decode as packed
T,H,HV,K,V=4,8,24,128,128
torch.manual_seed(19);dev='cuda';dt=torch.bfloat16
q=torch.randn(1,T,H,K,device=dev,dtype=dt);k=torch.randn_like(q);v=torch.randn(1,T,HV,V,device=dev,dtype=dt)
a=torch.randn(T,HV,device=dev,dtype=dt);b=torch.randn_like(a);A=torch.randn(HV,device=dev);bias=torch.randn_like(A)
records=[]
for sd in [torch.float32,torch.bfloat16]:
 initial=torch.randn(1,HV,V,K,device=dev,dtype=sd)*.1
 def run(n):
  state=initial.repeat(1+n*T,1,1,1);ids=torch.arange(1,1+n*T,device=dev,dtype=torch.int32).reshape(n,T)
  out,_=f(A,a.repeat(n,1),b.repeat(n,1),bias,q.repeat(1,n,1,1),k.repeat(1,n,1,1),v.repeat(1,n,1,1),initial_state=state,cu_seqlens=torch.arange(n+1,device=dev,dtype=torch.int32)*T,ssm_state_indices=ids,num_accepted_tokens=torch.ones(n,device=dev,dtype=torch.int32),use_qk_l2norm_in_kernel=True)
  return out,state
 u,us=run(1);vv,vs=run(4)
 records.append(dict(test='C1_C4',dtype=str(sd),output_equal=all(torch.equal(u,vv[:,i*T:(i+1)*T]) for i in range(4)),state_equal=all(torch.equal(us[1:],vs[1+i*T:1+(i+1)*T]) for i in range(4))))
 state=initial.repeat(2,1,1,1);outs=[];states=[]
 for i in range(T):
  o,_=f(A,a[i:i+1],b[i:i+1],bias,q[:,i:i+1],k[:,i:i+1],v[:,i:i+1],initial_state=state,cu_seqlens=torch.tensor([0,1],device=dev,dtype=torch.int32),ssm_state_indices=torch.tensor([1],device=dev,dtype=torch.int32),use_qk_l2norm_in_kernel=True)
  outs.append(o);states.append(state[1:].clone())
 seq=torch.cat(outs,dim=1);ss=torch.cat(states)
 records.append(dict(test='four_token_vs_single_steps',dtype=str(sd),output_equal=torch.equal(u,seq),state_equal=torch.equal(us[1:],ss),max_abs=float((u.float()-seq.float()).abs().max())))
 state=initial.repeat(2,1,1,1);outs=[];states=[]
 for i in range(T):
  x=torch.cat([q[0,i].reshape(-1),k[0,i].reshape(-1),v[0,i].reshape(-1)]).unsqueeze(0)
  o=torch.empty(1,1,HV,V,device=dev,dtype=dt)
  packed(x,a[i:i+1],b[i:i+1],A,bias,K**-.5,state,o,torch.tensor([1],device=dev,dtype=torch.int32),True)
  outs.append(o);states.append(state[1:].clone())
 seq=torch.cat(outs,dim=1);ss=torch.cat(states)
 records.append(dict(test='four_token_vs_packed_decode',dtype=str(sd),output_equal=torch.equal(u,seq),state_equal=torch.equal(us[1:],ss),max_abs=float((u.float()-seq.float()).abs().max())))
print(json.dumps(records,indent=2))
