import importlib.util,json,torch
from pathlib import Path
p=Path('/experiments/spec-determinism-20260914/fused_sigmoid.fixed.py');spec=importlib.util.spec_from_file_location('candidate_recurrence',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.manual_seed(917352);device='cuda';H,HV,K,V,T=8,24,128,128,6
q=torch.randn(1,T,H,K,device=device,dtype=torch.bfloat16);k=torch.randn_like(q);v=torch.randn(1,T,HV,V,device=device,dtype=torch.bfloat16);a=torch.randn(1,T,HV,device=device,dtype=torch.bfloat16);b=torch.randn_like(a);A=torch.randn(HV,device=device);bias=torch.randn(HV,device=device);initial=torch.randn(T+1,HV,V,K,device=device,dtype=torch.float32)*.01
f=m.fused_sigmoid_gating_delta_rule_update
rows=[]
for accepted in range(1,T+1):
 def run(N):
  state=torch.cat([initial[:1]]+[initial[1:]]*N).clone();indices=torch.arange(1,N*T+1,device=device,dtype=torch.int32).reshape(N,T)
  out,final=f(A,a.repeat(1,N,1),b.repeat(1,N,1),bias,q.repeat(1,N,1,1),k.repeat(1,N,1,1),v.repeat(1,N,1,1),initial_state=state,cu_seqlens=torch.arange(0,(N+1)*T,T,device=device,dtype=torch.int32),ssm_state_indices=indices,num_accepted_tokens=torch.full((N,),accepted,device=device,dtype=torch.int32),use_qk_l2norm_in_kernel=True)
  return out,final
 one,state=run(1);four,batch_state=run(4)
 assert all(torch.equal(one,four[:,i*T:(i+1)*T]) and torch.equal(state[1:],batch_state[1+i*T:1+(i+1)*T]) for i in range(4))
 single=torch.stack([initial[0],initial[accepted]]).clone()
 for t in range(T):
  out,_=f(A,a[:,t:t+1],b[:,t:t+1],bias,q[:,t:t+1],k[:,t:t+1],v[:,t:t+1],initial_state=single,cu_seqlens=torch.tensor([0,1],device=device,dtype=torch.int32),ssm_state_indices=torch.tensor([[1]],device=device,dtype=torch.int32),num_accepted_tokens=torch.ones(1,device=device,dtype=torch.int32),use_qk_l2norm_in_kernel=True)
  assert torch.equal(out,one[:,t:t+1]) and torch.equal(single[1],state[t+1])
 rows.append({'accepted':accepted,'six_token_C1_C4_exact':True,'six_single_steps_outputs_and_states_exact':True})
print(json.dumps({'passed':True,'rows':rows,'scope':'Actual fixed recurrent GPU kernel, six tokens, all accepted-state indices; full convolution/model/scheduler integration still pending.'},indent=2))
