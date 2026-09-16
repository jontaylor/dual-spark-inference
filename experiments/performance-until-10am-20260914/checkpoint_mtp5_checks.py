"""CPU tensor oracle for five-step accepted-state checkpoint selection."""
import importlib.util,sys,torch,types,json
from unittest.mock import patch
from vllm.v1.kv_cache_interface import MambaSpec,CircularBufferSpec
from vllm.model_executor.layers.mamba import mamba_utils as mu
spec=importlib.util.spec_from_file_location('aligned_completion_mtp5','/tmp/completion.aligned.mtp5.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
N=types.SimpleNamespace
results=[]
for dim_first in (True,False):
 for accepted in range(1,7):
  for delta in range(0,7):
   conv=torch.arange(7*2*8,dtype=torch.float32).reshape(7,2,8)
   if not dim_first:conv=conv.transpose(1,2).contiguous()
   ssm=torch.arange(7*3*2*2,dtype=torch.float32).reshape(7,3,2,2);ring=torch.arange(7*16,dtype=torch.uint8).reshape(7,16)
   ms=MambaSpec(1600,(tuple(conv.shape[1:]),(3,2,2)),(torch.float32,torch.float32));cs=CircularBufferSpec(8,num_kv_heads=1,head_size=2,dtype=torch.uint8)
   groups=[N(kv_cache_spec=ms,layer_names=['gdn']),N(kv_cache_spec=cs,layer_names=['ring'])]
   worker=N(completion_replacements={},kv_caches=N(group_data_refs=[[],[N(tensor_idx=0,page_size_bytes=16)]],tensors=[N(tensor=ring,page_size_bytes=16)]),tensor_offsets=[256],state_offset=lambda t,g:0 if t is conv else 128)
   connector=N(connector_worker=N(worker=worker),_completion_groups=groups,_completion_order=[0,1],_invalid_completions=set())
   runner=N(req_states=N(req_id_to_index={'r':0},num_computed_tokens=N(gpu=torch.tensor([64+delta]))),model_state=N(num_accepted_tokens_gpu=torch.tensor([accepted]),_mamba_state_idx_gpu=torch.tensor([0])),num_speculative_steps=5,model=N(get_mamba_state_copy_funcs=lambda kinds:{ms.mamba_type:[mu.get_conv_copy_spec,mu.get_temporal_copy_spec]}),vllm_config=N(compilation_config=N(static_forward_context={'gdn':N(kv_cache=(conv,ssm))})),block_tables=N(blocks_per_kv_block=[1,1],num_blocks=N(np=torch.tensor([[6],[1]]).numpy()),block_tables=[N(gpu=torch.tensor([[1,2,3,4,5,6]])),N(gpu=torch.tensor([[2]]))]))
   with patch.object(mu,'is_conv_state_dim_first',return_value=dim_first):
    m.capture_completion_states(connector,runner,m.CompletionMetadata(load_jobs={},store_jobs={},checkpoint_requests=[('r',64)]))
   # State after boundary is the accepted trajectory with delta suffix steps removed.
   index=accepted-1-delta
   if index<0:
    assert 'r' not in worker.aligned_completion_shadows
    results.append({'dim_first':dim_first,'accepted':accepted,'crossed_by':delta,'historical_state_missing_rejected':True});continue
   normalized=torch.zeros_like(conv[1]);axis=1 if dim_first else 0
   if dim_first:normalized[:,:8-index]=conv[1,:,index:]
   else:normalized[:8-index]=conv[1,index:]
   expected={0:[(0,normalized.view(torch.uint8).numpy().tobytes()),(128,ssm[1+index].view(torch.uint8).numpy().tobytes())],1:[(256,ring[2].numpy().tobytes())]}
   assert worker.aligned_completion_shadows['r']==(64,expected)
   conv.fill_(-1);ssm.fill_(-2);ring.fill_(255);runner.req_states.num_computed_tokens.gpu[0]=111
   save=m.CompletionSave(m.CompletionRecord(65,b'digest',64,[]),([1,2,3,4,5,6],[2]),100)
   m.capture_completion_states(connector,runner,m.CompletionMetadata(load_jobs={},store_jobs={},completion_saves={'r':save},checkpoint_finished={'r'}))
   assert worker.completion_replacements[100]==expected
   assert not worker.aligned_completion_shadows and not connector._invalid_completions
   results.append({'dim_first':dim_first,'accepted':accepted,'crossed_by':delta,'retained_exact_state':True})
print(json.dumps({'scope':'CPU state-selection/layout checks only, not GPU recurrence or model determinism','cases':results,'passed':len(results)},indent=2))
