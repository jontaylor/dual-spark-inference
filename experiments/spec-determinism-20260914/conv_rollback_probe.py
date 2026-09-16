import json,torch,importlib.util,sys
from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_update as conv
from vllm.third_party.flash_linear_attention.ops import fused_recurrent_gated_delta_rule_packed_decode as packed
spec=importlib.util.spec_from_file_location('spec_fixed','/tmp/fused_sigmoid.fixed.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);f=mod.fused_sigmoid_gating_delta_rule_update
T,H,HV,K,V=4,8,24,128,128;D=2*H*K+HV*V;dev='cuda';dt=torch.bfloat16
torch.manual_seed(23)
x=torch.randn(8,D,device=dev,dtype=dt);a=torch.randn(8,HV,device=dev,dtype=dt);b=torch.randn_like(a);A=torch.randn(HV,device=dev);bias=torch.randn_like(A);w=torch.randn(D,4,device=dev,dtype=dt)*.1
history=torch.randn(1,D,3,device=dev,dtype=dt);records=[]
for sd in [torch.float32,torch.bfloat16]:
 init=torch.randn(1,HV,V,K,device=dev,dtype=sd)*.1
 def plain(count):
  cs=history.repeat(2,1,1);ss=init.repeat(2,1,1,1);out=[]
  for t in range(count):
   xx=conv(x[t:t+1].clone(),cs,w,None,'silu',conv_state_indices=torch.tensor([1],device=dev,dtype=torch.int32),validate_data=False)
   o=torch.empty(1,1,HV,V,device=dev,dtype=dt);packed(xx,a[t:t+1],b[t:t+1],A,bias,K**-.5,ss,o,torch.tensor([1],device=dev,dtype=torch.int32),True);out.append(o)
  return torch.cat(out,dim=1),cs,ss
 def step(cs,ss,ids,start,accepted):
  n=ids.shape[0];cu=torch.arange(n+1,device=dev,dtype=torch.int32)*T
  xx=conv(x[start:start+T].repeat(n,1),cs,w,None,'silu',conv_state_indices=ids[:,0],num_accepted_tokens=torch.full((n,),accepted,device=dev,dtype=torch.int32),query_start_loc=cu,max_query_len=T,validate_data=False)
  q,k,v=torch.split(xx,[H*K,H*K,HV*V],dim=-1)
  out,_=f(A,a[start:start+T].repeat(n,1),b[start:start+T].repeat(n,1),bias,q.reshape(1,n*T,H,K),k.reshape(1,n*T,H,K),v.reshape(1,n*T,HV,V),initial_state=ss,cu_seqlens=cu,ssm_state_indices=ids,num_accepted_tokens=torch.full((n,),accepted,device=dev,dtype=torch.int32),use_qk_l2norm_in_kernel=True)
  return out
 def initial(n):
  cs=torch.zeros(1+n*T,D,3+T-1,device=dev,dtype=dt);cs[:,:,:3]=history
  return cs,init.repeat(1+n*T,1,1,1),torch.arange(1,1+n*T,device=dev,dtype=torch.int32).reshape(n,T)
 cs,ss,ids=initial(1);u=step(cs,ss,ids,0,1);ref,_,rs=plain(T)
 records.append(dict(test='conv_plus_recurrence_vs_packed',dtype=str(sd),output_equal=torch.equal(u,ref),state_equal=torch.equal(ss[4],rs[1])))
 cc,st,ii=initial(4);vv=step(cc,st,ii,0,1)
 records.append(dict(test='full_C1_C4',dtype=str(sd),output_equal=all(torch.equal(u,vv[:,j*T:(j+1)*T]) for j in range(4)),state_equal=all(torch.equal(ss[1:],st[1+j*T:1+(j+1)*T]) for j in range(4))))
 for accepted in range(1,T+1):
  c,s,i=initial(1);step(c,s,i,0,1);o=step(c,s,i,accepted,accepted)
  r,rc,rs=plain(accepted+T)
  records.append(dict(test='rejection_resume',accepted=accepted,dtype=str(sd),output_equal=torch.equal(o,r[:,accepted:]),state_equal=torch.equal(s[4],rs[1]),conv_equal=torch.equal(c[1,:,-3:],rc[1])))
print(json.dumps(records,indent=2));assert all(r['output_equal'] and r['state_equal'] and r.get('conv_equal',True) for r in records)
