import ast,json,types
from pathlib import Path
from vllm.v1.core.kv_cache_utils import mamba_aligned_replay_boundary
source=ast.parse(Path('/e/scheduler.fixed.py').read_text())
cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='Scheduler')
nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['_mamba_block_aligned_split','_cache_aligned_split']]
ns={'Request':object,'mamba_aligned_replay_boundary':mamba_aligned_replay_boundary}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<actual scheduler methods>','exec'),ns)
scheduler=types.SimpleNamespace(block_size=1600,cache_config=types.SimpleNamespace(block_size=1600),use_eagle=False,mamba_has_prefill_checkpoint_blocks=False,max_num_scheduled_tokens=128,_gb10_prefill_limit=0,mamba_partial_cache_hit=False,hash_block_size=1600)
for n in nodes:setattr(scheduler,n.name,types.MethodType(ns[n.name],scheduler))
request=types.SimpleNamespace(num_computed_tokens=0,num_prompt_tokens=193,num_tokens=193,shared_prefix_boundary=0)
parts=[]
while request.num_computed_tokens<193:
    remaining=193-request.num_computed_tokens
    n=scheduler._mamba_block_aligned_split(request,min(73,remaining));assert n>0
    parts.append(n);request.num_computed_tokens+=n
assert parts==[64,64,65],parts
scheduler.max_num_scheduled_tokens=8192
request.num_computed_tokens=0;request.num_prompt_tokens=request.num_tokens=1930
first=scheduler._mamba_block_aligned_split(request,1930);assert first==1600
request.num_computed_tokens=first
last=scheduler._mamba_block_aligned_split(request,1930-first);assert last==330
request.num_computed_tokens=1601
try:scheduler._mamba_block_aligned_split(request,329)
except ValueError:rejected=True
else:rejected=False
assert rejected
out={'193_token_prompt_budget73':parts,'cache_materialization_1930':[first,last],'unaligned_cached_state_rejected':rejected}
Path('/e/scheduler-checks.json').write_text(json.dumps(out,indent=2));print(out)
