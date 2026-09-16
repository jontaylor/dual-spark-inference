"""Real CUDA -> disk -> CUDA eviction fence oracle, isolated tiny test pages."""
import json
import sys
import time
from types import SimpleNamespace as N

import torch
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.kv_offload.base import (CanonicalKVCaches, CanonicalKVCacheRef,
    CanonicalKVCacheTensor, GPULoadStoreSpec)
from vllm.v1.kv_offload.rank_local_disk import DiskSlotManager, RankLocalDiskWorker
from vllm.v1.kv_offload.gb10_native_pressure import NativePressureCache
from vllm.distributed.kv_transfer.kv_connector.v1.gb10_completion import CompletionMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.gb10_aligned_offloading_connector import GB10AlignedOffloadingConnector
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import OffloadingConnectorWorker
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingWorkerMetadata

rank=int(sys.argv[1]);torch.cuda.set_device(0)
pool=BlockPool(8,True,64); manager=DiskSlotManager(8)
ids=iter(range(100,1000))
s=N(manager=manager,config=N(blocks_per_chunk=1,num_workers=2,
    kv_group_configs=[N(tokens_per_chunk=1920,group_idx=i) for i in range(2)]),
    _generate_job_id=lambda:next(ids))
c=N(connector_scheduler=s,_index={3:0,4:1},_completion_groups=[1,2,3])
cache=NativePressureCache(c,pool)
pages=[torch.arange(8*4096,device='cuda',dtype=torch.int32).remainder(251).to(torch.uint8).reshape(8,4096),
    torch.full((8,4096),41+rank,device='cuda',dtype=torch.uint8)]
layout=CanonicalKVCaches([CanonicalKVCacheTensor(t,4096) for t in pages],
    [[CanonicalKVCacheRef(0,4096)],[CanonicalKVCacheRef(1,4096)],[]])
worker=RankLocalDiskWorker(layout,'/tmp/native-pressure-oracle','test-'+str(time.time_ns()),rank,2,8192,verify_transfers=True)
ow=OffloadingConnectorWorker.__new__(OffloadingConnectorWorker)
ow.worker=worker;ow._unsubmitted_store_jobs=[];ow._is_store_writer=True
ow._connector_worker_meta=OffloadingWorkerMetadata();ow._load_jobs={}
connector=GB10AlignedOffloadingConnector.__new__(GB10AlignedOffloadingConnector)
connector.connector_worker=ow;connector._completion_enabled=False
try:
    b=pool.get_new_blocks(2)
    expected=[]
    for i,block in enumerate(b):
        pages[i][block.block_id].add_(rank+i)
        expected.append(pages[i][block.block_id].clone())
        pool._insert_block_hash(make_block_hash_with_group_id(bytes([20+i])*32,3+i),block,1920)
    pool.free_blocks(b)
    cache.add_metadata(CompletionMetadata(load_jobs={},store_jobs={}))
    pool.get_new_blocks(5);assert not cache.jobs
    pool.get_new_blocks(2)
    meta=CompletionMetadata(load_jobs={},store_jobs={});cache.add_metadata(meta)
    jobs=dict(meta.store_jobs)
    assert len(jobs)==2 and not manager.resident
    connector.prepare_completion(N(),meta)
    # Simulate immediate reuse/zeroing by the new requests after the fence.
    for tensor in pages:tensor.fill_(177)
    results=worker.get_finished();assert len(results)==2
    assert all(r.success and r.transfer_size==8192 for r in results)
    completed={r.job_id:1 for r in results}
    cache.consume_completions(N(completed_jobs=completed))
    assert len(cache.jobs)==2
    cache.consume_completions(N(completed_jobs=completed))
    assert not cache.jobs
    load_ids=[]
    for index,(jid,job) in enumerate(jobs.items()):
        load_id=jid+1000;load_ids.append(load_id)
        dst=GPULoadStoreSpec([6+index],job.src_spec.group_sizes,job.src_spec.block_indices)
        worker.submit_load(load_id,job.dst_spec,dst)
    worker.wait(set(load_ids))
    loaded=worker.get_finished()
    assert len(loaded)==2 and all(r.success and r.transfer_size==8192 for r in loaded)
    for i in range(2):
        assert torch.equal(pages[i][6+i],expected[i])
        assert bool(torch.all(pages[1-i][6+i]==177))
    print(json.dumps({'passed':True,'rank':rank,'saved_blocks':2,'restored_blocks':2,
        'store_bytes':sum(r.transfer_size for r in results),'load_bytes':sum(r.transfer_size for r in loaded),
        'source_overwritten_after_fence':True,'restored_bytes_exact':True,
        'unrelated_group_unchanged':True,'gpu_readback_verification':True}),flush=True)
finally:
    worker.shutdown()
