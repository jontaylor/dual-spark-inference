"""CPU state-selection and scheduling gates, no inference mutation."""
import ast,importlib.util,sys,torch,types,json
from pathlib import Path
from unittest.mock import patch
from vllm.v1.kv_cache_interface import MambaSpec,CircularBufferSpec
from vllm.model_executor.layers.mamba import mamba_utils as mu
spec=importlib.util.spec_from_file_location('event_completion','/tmp/completion.events.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
N=types.SimpleNamespace
# Ordinary decode metadata must not dereference model state, even if a stale
# caller supplies historical checkpoint requests.
m.capture_completion_states(N(),N(),m.CompletionMetadata(load_jobs={},store_jobs={},checkpoint_requests=[('r',64)]))
passed=0
for dim_first in (True,False):
 for accepted in range(1,5):
  for delta in range(5):
   conv=torch.arange(6*2*8,dtype=torch.float32).reshape(6,2,8)
   if not dim_first:conv=conv.transpose(1,2).contiguous()
   ssm=torch.arange(6*3*2*2,dtype=torch.float32).reshape(6,3,2,2)
   ms=MambaSpec(1920,(tuple(conv.shape[1:]),(3,2,2)),(torch.float32,torch.float32))
   groups=[N(kv_cache_spec=ms,layer_names=['gdn'])]
   worker=N(completion_replacements={},state_offset=lambda t,g:0 if t is conv else 128)
   connector=N(connector_worker=N(worker=worker),_completion_groups=groups,_invalid_completions=set())
   runner=N(req_states=N(req_id_to_index={'r':0},num_computed_tokens=N(gpu=torch.tensor([137+delta]))),model_state=N(num_accepted_tokens_gpu=torch.tensor([accepted]),_mamba_state_idx_gpu=torch.tensor([0])),num_speculative_steps=3,model=N(get_mamba_state_copy_funcs=lambda kinds:{ms.mamba_type:[mu.get_conv_copy_spec,mu.get_temporal_copy_spec]}),vllm_config=N(compilation_config=N(static_forward_context={'gdn':N(kv_cache=(conv,ssm))})))
   save=m.CompletionSave(m.CompletionRecord(138,b'digest',137,[]),([1,2,3,4],),100)
   with patch.object(mu,'is_conv_state_dim_first',return_value=dim_first):
    m.capture_completion_states(connector,runner,m.CompletionMetadata(load_jobs={},store_jobs={},completion_saves={'r':save}))
   offset=accepted-1-delta
   if offset<0:
    assert 100 in connector._invalid_completions and 100 not in worker.completion_replacements
   else:
    norm=torch.zeros_like(conv[1])
    if dim_first:norm[:,:8-offset]=conv[1,:,offset:]
    else:norm[:8-offset]=conv[1,offset:]
    assert worker.completion_replacements[100]=={0:[(0,norm.view(torch.uint8).numpy().tobytes()),(128,ssm[1+offset].view(torch.uint8).numpy().tobytes())]}
   passed+=1
# Exercise actual new scheduler method without importing a GPU-serving scheduler.
tree=ast.parse(Path('/tmp/scheduler.events.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Scheduler');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_mamba_block_aligned_split');scope={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'candidate','exec'),scope)
f=scope[fn.name];cases=0
for offset in range(1,64):
 for remaining in (1,10,100,500):
  start=1920+offset;request=N(num_computed_tokens=start,num_prompt_tokens=start+remaining,num_tokens=start+remaining)
  for budget in (1,16,32,64,8192):
   result=f(N(),request,budget);required=min(64-offset,remaining)
   assert result==(required if budget>=required else 0)
   cases+=1
print(json.dumps({'state_selection_cases':passed,'partial_grid_cases':cases,'ordinary_decode_capture_noop':True,'passed':True}))
