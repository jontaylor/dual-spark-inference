"""Actual hybrid multi-group preflight must defer before destination allocation."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as N
import torch
P=Path(__file__).parent
def load(name,file):
    spec=importlib.util.spec_from_file_location(name,P/file)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
load('vllm.v1.kv_offload.gb10_allocation_deferral','deferral.py')
common=load('vllm.distributed.kv_transfer.kv_connector.v1.offloading.common','common.py')
load('vllm.v1.core.block_pool','block_pool.py')
d=load('vllm.v1.kv_offload.rank_local_disk','rank_local_disk.py')
n=load('vllm.v1.kv_offload.gb10_native_pressure','native_pressure.py')
k=load('vllm.v1.core.kv_cache_manager','kv_cache_manager.py')
from vllm.v1.kv_offload.gb10_allocation_deferral import AllocationDeferred
from vllm.v1.core.sched.parking_policy import ParkingPolicy
from vllm.v1.kv_cache_interface import FullAttentionSpec,MambaSpec,KVCacheGroupSpec,KVCacheConfig
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.request import Request
from vllm import SamplingParams
att=FullAttentionSpec(block_size=1920,num_kv_heads=1,head_size=256,dtype=torch.bfloat16)
rec=MambaSpec(block_size=1920,shapes=((1920,1,256),),dtypes=(torch.bfloat16,),mamba_cache_mode='align',num_speculative_blocks=3)
kv=KVCacheConfig(num_blocks=20,kv_cache_tensors=[],kv_cache_groups=[
    KVCacheGroupSpec(layer_names=['attention'],kv_cache_spec=att),
    KVCacheGroupSpec(layer_names=['recurrent'],kv_cache_spec=rec)])
cache=k.KVCacheManager(kv,max_model_len=7680,scheduler_block_size=1920,hash_block_size=1920,use_eagle=True)
pool=cache.block_pool;manager=d.DiskSlotManager(100);manager.configure_memory(pool,ParkingPolicy(19));manager.native_mode=True
ids=iter(range(100,10000));sched=N(manager=manager,config=N(blocks_per_chunk=1,num_workers=2),_generate_job_id=lambda:next(ids))
native=n.NativePressureCache(N(connector_scheduler=sched,_completion_groups=[1,2]),pool)
old=pool.get_new_blocks(19);keys=[str(b.block_id).encode() for b in old];ctx=ReqContext('old')
manager.prepare_memory_store(keys,ctx,{key:(0,i) for i,key in enumerate(keys)},borrowed={key:b.block_id for key,b in zip(keys,old)})
manager.complete_store(keys,ctx);pool.free_blocks(old)
r=Request(request_id='new',prompt_token_ids=[1,2,3,4],sampling_params=SamplingParams(max_tokens=16),pooling_params=None)
try:cache.allocate_slots(r,4,num_lookahead_tokens=3)
except AllocationDeferred:pass
else:raise AssertionError('hybrid allocation did not defer')
assert all(not m.req_to_blocks.get('new') for m in cache.coordinator.single_type_managers)
assert r.num_computed_tokens==0 and native.source_blocks
jobs=set(native.jobs)
for rank in (0,1):native.consume_completions(common.OffloadingWorkerMetadata(source_preserved={j:{rank} for j in jobs}))
result=cache.allocate_slots(r,4,num_lookahead_tokens=3)
assert result is not None and all(len(g)>0 for g in result.blocks)
assert all(not b.is_null and b.ref_cnt>0 for g in result.blocks for b in g)
assert not native.source_blocks
assert manager.spilling, 'disk gate released too early'
print(json.dumps({'passed':True,'groups':[len(g) for g in result.blocks],
    'checks':['all hybrid groups unallocated on deferral','MTP3 lookahead allocation',
              'two-rank source ACK permits complete retry','disk still pending after retry']}))
