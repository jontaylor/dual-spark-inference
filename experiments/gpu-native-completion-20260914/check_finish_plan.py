"""Real completion planner: repeated turns share attention and allocate only state."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as N
import torch

p=Path(__file__).resolve().parent
def load(name,file):
    spec=importlib.util.spec_from_file_location(name,p/file)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
load('vllm.v1.core.block_pool','block_pool.gpu.py')
d=load('vllm.v1.kv_offload.rank_local_disk','rank_local_disk.gpu.py')
c=load('vllm.distributed.kv_transfer.kv_connector.v1.gb10_completion','completion.gpu.py')
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.parking_policy import ParkingPolicy
from vllm.v1.kv_cache_interface import FullAttentionSpec,MambaSpec
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.request import RequestStatus
pool=BlockPool(40,True,1920);ledger=ParkingPolicy(39)
manager=d.DiskSlotManager(80);manager.configure_memory(pool,ledger);manager.native_mode=True
att=FullAttentionSpec(block_size=1920,num_kv_heads=1,head_size=256,dtype=torch.bfloat16)
rec=MambaSpec(block_size=1920,shapes=((6,16),(2,8,8)),dtypes=(torch.bfloat16,torch.float32),
              mamba_cache_mode='align',num_speculative_blocks=3)
groups=[N(kv_cache_spec=att),N(kv_cache_spec=rec)]
counter=iter(range(100,1000))
s=N(manager=manager,_req_status={},_jobs={},_generate_job_id=lambda:next(counter),
    config=N(num_workers=2,kv_group_configs=[N(sliding_window_size_in_chunks=None,is_eagle_group=True),
                                            N(sliding_window_size_in_chunks=None,is_eagle_group=False)]))
connector=N(connector_scheduler=s,_alignment=1920)
cache=c.CompletionCache(connector,groups)
attention=pool.get_new_blocks(3)
for turn in range(2):
    rid=f'producer-{turn}'
    if turn:pool.touch(attention)
    source=pool.get_new_blocks(4)
    state=N(req_context=ReqContext(rid),group_states=[N(offload_keys=[]),N(offload_keys=[])],
            update_offload_keys=lambda:None,transfer_jobs=set())
    s._req_status[rid]=state
    request=N(request_id=rid,num_computed_tokens=4100+turn*10,num_tokens=4101+turn*10,
              all_token_ids=list(range(4101+turn*10)),cache_salt=None,mm_features=[],prompt_embeds=None,
              lora_request=None,skip_reading_prefix_cache=False,num_in_flight_tokens=0,
              status=RequestStatus.FINISHED_LENGTH_CAPPED)
    free_before=pool.get_num_free_blocks()
    assert cache.finish(request,([b.block_id for b in attention],[0,0]+[b.block_id for b in source]),memory=True)
    save,shared,context=cache.pending[rid]; job=cache.to_send[rid][1]
    assert len(shared)==3*turn
    assert len(job.dst_spec.memory_blocks)==4-3*turn
    assert pool.get_num_free_blocks()==free_before-1 and ledger.cache_reserved==1
    keys=list(s._jobs[save.job_id].keys) if hasattr(s._jobs[save.job_id],'keys') else [k for _,_,k in save.record.pages if k not in shared]
    manager.complete_store(keys,context);state.transfer_jobs.clear()
    cache.update(N(kv_connector_worker_meta=N(invalid_completions=set()),finished_sending=None))
    pool.free_blocks(attention+source)
    assert ledger.cache_reserved==0 and pool.get_num_free_blocks()==39
assert len(manager.resident)==5  # three shared attention pages + one state per turn
assert len(cache.records)==2
# A pressure suspension must not inherit GPU-only pages and pin that memory.
rid='parked';pool.touch(attention);source=pool.get_new_blocks(4)
state=N(req_context=ReqContext(rid),group_states=[N(offload_keys=[]),N(offload_keys=[])],
        update_offload_keys=lambda:None,transfer_jobs=set())
s._req_status[rid]=state
request=N(request_id=rid,num_computed_tokens=4130,num_tokens=4131,
          all_token_ids=list(range(4131)),cache_salt=None,mm_features=[],prompt_embeds=None,
          lora_request=None,skip_reading_prefix_cache=False,num_in_flight_tokens=0,
          status=RequestStatus.RUNNING)
record=next(reversed(cache.records.values()))
cache.restored_pages[rid]={(gi,index):key for gi,index,key in record.pages if gi==0}
assert cache.finish(request,([b.block_id for b in attention],[0,0]+[b.block_id for b in source]),active=True)
save,shared,context=cache.pending[rid];job=cache.to_send[rid][1]
assert not shared and len(job.dst_spec.block_ids)==4
assert not isinstance(job.dst_spec,d.ResidentSlots)
print(json.dumps({'passed':True,'turns':2,'attention_pages':3,'physical_completion_pages':5,
                  'attention_pages_copied':0,'state_pages_allocated_per_turn':1,
                  'all_completed_pages_evictable':True,'parked_state_requires_no_gpu_pins':True}),flush=True)
