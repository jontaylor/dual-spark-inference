import importlib.util,sys,torch,types,json,pickle
from unittest.mock import patch
from vllm.v1.kv_cache_interface import MambaSpec,CircularBufferSpec
from vllm.model_executor.layers.mamba import mamba_utils as mu
spec=importlib.util.spec_from_file_location('aligned_completion_test','/tmp/completion.aligned.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
N=types.SimpleNamespace
conv=torch.arange(5*2*6,dtype=torch.float32).reshape(5,2,6);ssm=torch.arange(5*3*2*2,dtype=torch.float32).reshape(5,3,2,2);ring=torch.arange(5*16,dtype=torch.uint8).reshape(5,16)
ms=MambaSpec(1600,((2,6),(3,2,2)),(torch.float32,torch.float32));cs=CircularBufferSpec(8,num_kv_heads=1,head_size=2,dtype=torch.uint8)
groups=[N(kv_cache_spec=ms,layer_names=['gdn']),N(kv_cache_spec=cs,layer_names=['ring'])]
worker=N(completion_replacements={},kv_caches=N(group_data_refs=[[],[N(tensor_idx=0,page_size_bytes=16)]],tensors=[N(tensor=ring,page_size_bytes=16)]),tensor_offsets=[256],state_offset=lambda t,g:0 if t is conv else 128)
connector=N(connector_worker=N(worker=worker),_completion_groups=groups,_completion_order=[0,1],_invalid_completions=set())
runner=N(req_states=N(req_id_to_index={'r':0},num_computed_tokens=N(gpu=torch.tensor([64]))),model_state=N(num_accepted_tokens_gpu=torch.tensor([2]),_mamba_state_idx_gpu=torch.tensor([0])),num_speculative_steps=3,model=N(get_mamba_state_copy_funcs=lambda kinds:{ms.mamba_type:[mu.get_conv_copy_spec,mu.get_temporal_copy_spec]}),vllm_config=N(compilation_config=N(static_forward_context={'gdn':N(kv_cache=(conv,ssm))})),block_tables=N(blocks_per_kv_block=[1,1],num_blocks=N(np=torch.tensor([[4],[1]]).numpy()),block_tables=[N(gpu=torch.tensor([[1,2,3,4]])),N(gpu=torch.tensor([[2]]))]))
expected_conv=torch.zeros_like(conv[1]);expected_conv[:,:5]=conv[1,:,1:];expected={0:[(0,expected_conv.view(torch.uint8).numpy().tobytes()),(128,ssm[2].view(torch.uint8).numpy().tobytes())],1:[(256,ring[2].numpy().tobytes())]}
with patch.object(mu,'is_conv_state_dim_first',return_value=True):
 m.capture_completion_states(connector,runner,pickle.loads(pickle.dumps(m.CompletionMetadata(load_jobs={}, store_jobs={}, checkpoint_requests=[('r',64)]))))
 assert worker.aligned_completion_shadows['r']==(64,expected)
 conv.fill_(-7);ssm.fill_(-9);ring.fill_(255);runner.req_states.num_computed_tokens.gpu[0]=95
 save=m.CompletionSave(m.CompletionRecord(65,b'digest',64,[]),([4,3,2,1],[2]),100)
 m.capture_completion_states(connector,runner,m.CompletionMetadata(load_jobs={}, store_jobs={}, completion_saves={'r':save},checkpoint_finished={'r'}))
 assert worker.completion_replacements[100]==expected
 assert not worker.aligned_completion_shadows and not connector._invalid_completions
 # Missing historical state must be rejected instead of reusing current state.
 m.capture_completion_states(connector,runner,m.CompletionMetadata(load_jobs={}, store_jobs={}, completion_saves={'r':save}))
 assert 100 in connector._invalid_completions
print(json.dumps({'speculative_state_normalized_at_checkpoint':True,'ring_checkpoint_retained':True,'later_mutation_does_not_change_snapshot':True,'finished_request_memory_released':True,'missing_historical_state_rejected':True,'metadata_pickle_roundtrip':True},indent=2))
