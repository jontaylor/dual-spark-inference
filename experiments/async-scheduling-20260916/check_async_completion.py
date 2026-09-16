"""Actual completion planner retries allocation without dropping provenance."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as N
import torch
P = Path(sys.argv[1])
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import common
from vllm.v1.core import block_pool as bp
from vllm.v1.kv_offload import rank_local_disk as d
from vllm.v1.kv_offload import gb10_native_pressure as n
name='vllm.distributed.kv_transfer.kv_connector.v1.gb10_completion'
spec=importlib.util.spec_from_file_location(name,P/'completion.py')
c=importlib.util.module_from_spec(spec);sys.modules[name]=c;spec.loader.exec_module(c)
from vllm.v1.core.sched.parking_policy import ParkingPolicy
from vllm.v1.kv_cache_interface import FullAttentionSpec,MambaSpec,CircularBufferSpec
from vllm.v1.kv_offload.base import ReqContext,LookupResult
from vllm.v1.request import RequestStatus
pool=bp.BlockPool(14,True,1920);ledger=ParkingPolicy(13)
manager=d.DiskSlotManager(100);manager.configure_memory(pool,ledger);manager.native_mode=True
att=FullAttentionSpec(block_size=1920,num_kv_heads=1,head_size=256,dtype=torch.bfloat16)
rec=MambaSpec(block_size=1920,shapes=((6,16),(2,8,8)),dtypes=(torch.bfloat16,torch.float32),
             mamba_cache_mode='align',num_speculative_blocks=3)
ring_spec=CircularBufferSpec(block_size=8,num_kv_heads=1,head_size=2,dtype=torch.uint8)
groups=[N(kv_cache_spec=att),N(kv_cache_spec=rec),N(kv_cache_spec=ring_spec)]
counter=iter(range(100,1000))
s=N(manager=manager,_req_status={},_jobs={},_generate_job_id=lambda:next(counter),
    config=N(blocks_per_chunk=1,num_workers=2,kv_group_configs=[
        N(sliding_window_size_in_chunks=None,is_eagle_group=True),
        N(sliding_window_size_in_chunks=None,is_eagle_group=False),
        N(sliding_window_size_in_chunks=None,is_eagle_group=False)]))
connector=N(connector_scheduler=s,_alignment=1920,_completion_groups=groups,
            supports_versioned_completions=True, terminal_version=lambda rid: 41)
native=n.NativePressureCache(connector,pool);cache=c.CompletionCache(connector,groups)
old=pool.get_new_blocks(1)[0];ctx=ReqContext('old')
manager.prepare_memory_store([b'old'],ctx,{b'old':(0,0)},borrowed={b'old':old.block_id})
manager.complete_store([b'old'],ctx);pool.free_blocks([old])
attention=pool.get_new_blocks(3);source=pool.get_new_blocks(4);ring=pool.get_new_blocks(1);filler=pool.get_new_blocks(3)
rid='finishing'
state=N(req_context=ReqContext(rid),group_states=[N(offload_keys=[]),N(offload_keys=[]),N(offload_keys=[])],
        update_offload_keys=lambda:None,transfer_jobs=set(),finished_signaled=True)
s._req_status[rid]=state
request=N(request_id=rid,num_computed_tokens=4104,num_tokens=4101,all_token_ids=list(range(4101)),
          cache_salt=None,mm_features=[],prompt_embeds=None,lora_request=None,
          skip_reading_prefix_cache=False,num_in_flight_tokens=4,status=RequestStatus.FINISHED_LENGTH_CAPPED)
blocks=([b.block_id for b in attention],[0,0]+[b.block_id for b in source],[ring[0].block_id])
cache.restored_pages[rid]={}  # remains alive across repeated deferred attempts
assert cache.finish(request,blocks,memory=True)
assert rid in cache.deferred_allocations and rid not in cache.pending
assert state.allocation_pending and not s._jobs
assert all(b.ref_cnt==1 for b in attention+source)
assert rid in cache.restored_pages
cache.retry_allocations()
assert rid in cache.deferred_allocations and not s._jobs
jobs=set(native.jobs)
for rank in (0,1):
    native.consume_completions(common.OffloadingWorkerMetadata(source_preserved={j:{rank} for j in jobs}))
assert manager.lookup(b'old',ctx)==LookupResult.HIT_PENDING
cache.retry_allocations()
assert rid not in cache.deferred_allocations and rid in cache.pending
assert not state.allocation_pending and len(s._jobs)==1
save,shared,context=cache.pending[rid];job=cache.to_send[rid][1]
assert isinstance(job.dst_spec,d.ResidentSlots), 'temporary wait fell back to disk'
assert len(job.dst_spec.memory_blocks)==5
assert save.terminal_version==41 and save.request_id==rid and save.record.boundary==4100
assert job.dst_spec.memory_blocks[-1] != ring[0].block_id, "live ring was borrowed instead of copied"
assert pool.get_num_free_blocks()==0
keys=s._jobs[save.job_id].keys
manager.complete_store(keys,context);state.transfer_jobs.clear()
output=N(kv_connector_worker_meta=N(invalid_completions=set()),finished_sending=None)
cache.update(output)
assert output.finished_sending=={rid} and rid not in cache.pending
output.finished_sending=None;cache.update(output)
assert not output.finished_sending, 'duplicate finished notification'
assert all(b.ref_cnt==1 for b in attention+source)
print(json.dumps({'passed':True,'checks':['terminal output with later batch in flight', 'ring destination is private', 'immutable version preserved across retry', 'repeated completion deferral',
    'source/provenance retained','no extra transfer jobs while deferred',
    'two-rank copy ACK unblocks allocation before persistence',
    'memory retry rather than disk fallback','single finished notification']}))
