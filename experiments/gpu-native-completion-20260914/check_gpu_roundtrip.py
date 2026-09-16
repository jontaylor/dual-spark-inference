"""Real GPU completion save/load, forced overwrite, and disk restore oracle."""
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace as N

import torch

ROOT = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


load('vllm.v1.core.block_pool', 'block_pool.gpu.py')
disk = load('vllm.v1.kv_offload.rank_local_disk', 'rank_local_disk.gpu.py')
load('vllm.v1.kv_offload.gb10_gpu_checkpoint', 'gpu_checkpoint.py')
gpu_copy = load('vllm.v1.kv_offload.gb10_gpu_completion_copy', 'gpu_completion_copy.py')
completion = load('vllm.distributed.kv_transfer.kv_connector.v1.gb10_completion', 'completion.gpu.py')
pressure = load('vllm.v1.kv_offload.gb10_native_pressure', 'native_pressure.gpu.py')
connector_module = load('vllm.distributed.kv_transfer.kv_connector.v1.gb10_aligned_offloading_connector', 'connector.gpu.py')
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.parking_policy import ParkingPolicy
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches, CanonicalKVCacheTensor, CanonicalKVCacheRef,
    GPULoadStoreSpec, ReqContext, LookupResult,
)
from vllm.v1.kv_cache_interface import MambaSpec, FullAttentionSpec
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec, get_temporal_copy_spec, is_conv_state_dim_first,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import TransferJob, OffloadingWorkerMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import OffloadingConnectorWorker

rank = int(sys.argv[1]); torch.cuda.set_device(0)
pool = BlockPool(16, True, 1920); ledger = ParkingPolicy(15)
manager = disk.DiskSlotManager(24); manager.configure_memory(pool, ledger); manager.native_mode = True
ids = iter(range(100, 1000))
s = N(manager=manager, config=N(blocks_per_chunk=1, num_workers=2),
      _generate_job_id=lambda: next(ids))
conn = N(connector_scheduler=s, _completion_groups=[1, 2, 3])
cache = pressure.NativePressureCache(conn, pool)
conv_shape = (16, 6) if is_conv_state_dim_first() else (6, 16)
attention = torch.arange(16*1024, device='cuda', dtype=torch.int32).remainder(251).to(torch.uint8).reshape(16, 1024)
conv = torch.arange(16*96, device='cuda', dtype=torch.int32).reshape(16, *conv_shape).to(torch.bfloat16)
temporal = torch.arange(16*128, device='cuda', dtype=torch.float32).reshape(16, 2, 8, 8)
ring = torch.full((16, 512), 31+rank, device='cuda', dtype=torch.uint8)
tensors = [attention, conv, temporal, ring]
sizes = [t[0].numel()*t.element_size() for t in tensors]
layout = CanonicalKVCaches([CanonicalKVCacheTensor(t, size) for t, size in zip(tensors, sizes)],
                          [[CanonicalKVCacheRef(0, sizes[0])],
                           [CanonicalKVCacheRef(1, sizes[1]), CanonicalKVCacheRef(2, sizes[2])],
                           [CanonicalKVCacheRef(3, sizes[3])]])
worker = disk.RankLocalDiskWorker(layout, '/tmp/gpu-native-oracle', 'test-'+str(time.time_ns()),
                                  rank, 2, 4096, verify_transfers=True)
ow = OffloadingConnectorWorker.__new__(OffloadingConnectorWorker)
ow.worker=worker; ow._unsubmitted_store_jobs=[]; ow._is_store_writer=True
ow._connector_worker_meta=OffloadingWorkerMetadata(); ow._load_jobs={}
bridge = connector_module.GB10AlignedOffloadingConnector.__new__(connector_module.GB10AlignedOffloadingConnector)
bridge.connector_worker=ow; bridge._completion_enabled=False
try:
    sources = pool.get_new_blocks(6)
    att_id, ring_id = sources[0].block_id, sources[1].block_id
    recurrent_ids = [b.block_id for b in sources[2:]]
    keys=[b'att', b'rec', b'ring']; ctx=ReqContext('producer')
    result=manager.prepare_memory_store(keys, ctx, dict(zip(keys, [(0,0),(1,0),(2,0)])),
                                        borrowed={keys[0]:att_id,keys[2]:ring_id})
    gpu=GPULoadStoreSpec([att_id,recurrent_ids[-1],ring_id],[1,1,1],[0,0,0])
    jid=next(ids)
    spec=MambaSpec(block_size=1920,shapes=(conv_shape,(2,8,8)),
                   dtypes=(torch.bfloat16,torch.float32),mamba_cache_mode='align',num_speculative_blocks=3)
    att_spec=FullAttentionSpec(block_size=1920,num_kv_heads=1,head_size=256,dtype=torch.bfloat16)
    groups=[N(kv_cache_spec=att_spec,layer_names=[]),N(kv_cache_spec=spec,layer_names=['rec']),
            N(kv_cache_spec=att_spec,layer_names=[])]
    scalar=lambda x: torch.tensor([x],device='cuda',dtype=torch.int32)
    runner=N(req_states=N(req_id_to_index={'producer':0},num_computed_tokens=N(gpu=scalar(128))),
             model_state=N(num_accepted_tokens_gpu=scalar(3),_mamba_state_idx_gpu=scalar(0)),
             model=N(get_mamba_state_copy_funcs=lambda types:{spec.mamba_type:(get_conv_copy_spec,get_temporal_copy_spec)}),
             vllm_config=N(compilation_config=N(static_forward_context={'rec':N(kv_cache=(conv,temporal))})),
             num_speculative_steps=3)
    save=completion.CompletionSave(completion.CompletionRecord(129,b'digest',128,[]),
                                    ([att_id],recurrent_ids,[ring_id]),jid)
    meta=completion.CompletionMetadata(load_jobs={},store_jobs={jid:TransferJob('producer',gpu,result.store_spec)},
                                        completion_saves={'producer':save})
    fake=N(connector_worker=N(worker=worker),_completion_groups=groups,_invalid_completions=set())
    expected_conv=torch.zeros_like(conv[0])
    if is_conv_state_dim_first():expected_conv[:,:4].copy_(conv[recurrent_ids[0]][:,2:])
    else:expected_conv[:4].copy_(conv[recurrent_ids[0]][2:])
    expected=[attention[att_id].clone(),expected_conv,temporal[recurrent_ids[2]].clone(),ring[ring_id].clone()]
    completion.capture_completion_states(fake,runner,meta)
    assert not fake._invalid_completions
    worker.submit_store(jid,gpu,result.store_spec);worker.wait({jid})
    done=worker.get_finished();assert len(done)==1 and done[0].success and done[0].transfer_size==0
    manager.complete_store(keys,ctx);pool.free_blocks(sources)
    assert ledger.cache_reserved==0 and pool.get_num_free_blocks()==15
    # Only the normalized recurrent state owns a new physical page.
    assert result.store_spec.memory_blocks[0]==att_id and result.store_spec.memory_blocks[2]==ring_id
    cached_ids=list(result.store_spec.memory_blocks)
    assert torch.equal(conv[cached_ids[1]],expected[1]) and torch.equal(temporal[cached_ids[1]],expected[2])
    load_spec=manager.prepare_load(keys,ctx)
    targets=pool.get_new_blocks(3); dst=GPULoadStoreSpec([b.block_id for b in targets],[1,1,1],[0,0,0])
    load_jid=next(ids);worker.submit_load(load_jid,load_spec,dst);worker.wait({load_jid})
    loaded=worker.get_finished();assert len(loaded)==1 and loaded[0].transfer_size==0
    for tensor,index,oracle in [(attention,0,expected[0]),(conv,1,expected[1]),(temporal,1,expected[2]),(ring,2,expected[3])]:
        assert torch.equal(tensor[targets[index].block_id],oracle)
    manager.complete_load(keys,ctx);pool.free_blocks(targets)
    # No file was needed to save or recall the GPU completion.
    assert not list(worker.root.glob('slot-*.bin')) and not cache.jobs
    cache.add_metadata(completion.CompletionMetadata(load_jobs={},store_jobs={}))
    allocated=pool.get_new_blocks(15)
    assert len(cache.jobs)==3
    eviction_meta=completion.CompletionMetadata(load_jobs={},store_jobs={});cache.add_metadata(eviction_meta)
    bridge.prepare_completion(N(),eviction_meta)
    for tensor in tensors:tensor.fill_(177)  # actual destructive reuse after the early fence
    stored=worker.get_finished();assert len(stored)==3 and all(r.success for r in stored)
    ack=N(completed_jobs={r.job_id:1 for r in stored})
    cache.consume_completions(ack);assert len(cache.jobs)==3
    cache.consume_completions(ack);assert not cache.jobs and not manager.resident
    assert all(b.ref_cnt==1 for b in allocated)  # no accidental free of new owners
    disk_src=manager.prepare_load(keys,ctx)
    assert not isinstance(disk_src,disk.ResidentSlots)
    restore_id=next(ids);restore=GPULoadStoreSpec([12,13,14],[1,1,1],[0,0,0])
    worker.submit_load(restore_id,disk_src,restore);worker.wait({restore_id})
    restored=worker.get_finished();assert len(restored)==1 and restored[0].success
    for tensor,index,oracle in [(attention,12,expected[0]),(conv,13,expected[1]),(temporal,13,expected[2]),(ring,14,expected[3])]:
        assert torch.equal(tensor[index],oracle)
    manager.complete_load(keys,ctx)
    print(json.dumps({'passed':True,'rank':rank,'gpu_save_bytes_to_disk':0,'gpu_recall_bytes_from_disk':0,
                      'attention_and_ring_shared':True,'accepted_recurrent_state_exact':True,
                      'zero_permanent_reservation':True,'overwrite_after_eviction_fence':True,
                      'disk_restore_exact':True,'gpu_readback_verified':True,
                      'store_bytes_after_actual_reuse':sum(r.transfer_size for r in stored)}),flush=True)
finally:
    worker.shutdown()
