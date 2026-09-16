"""Import candidate as a unit and prove reserved jobs never enter worker.wait."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as N
P=Path(__file__).parent
def load(name,file):
    spec=importlib.util.spec_from_file_location(name,P/file)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
load('vllm.v1.kv_offload.gb10_allocation_deferral','deferral.py')
common=load('vllm.distributed.kv_transfer.kv_connector.v1.offloading.common','common.py')
load('vllm.v1.core.block_pool','block_pool.py')
load('vllm.v1.core.kv_cache_manager','kv_cache_manager.py')
load('vllm.v1.core.sched.scheduler','scheduler.py')
load('vllm.v1.kv_offload.rank_local_disk','rank_local_disk.py')
load('vllm.v1.kv_offload.gb10_native_pressure','native_pressure.py')
s=load('vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler','offloading_scheduler.py')
load('vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker','worker.py')
c=load('vllm.distributed.kv_transfer.kv_connector.v1.gb10_completion','completion.py')
connector=load('vllm.distributed.kv_transfer.kv_connector.v1.gb10_aligned_offloading_connector','connector.py')
load('vllm.v1.core.sched.gb10_parking_scheduler','parking_scheduler.py')
# Real slotted request state must support the pending allocation flag.
state=object.__new__(s.RequestOffloadState);state.allocation_pending=True
assert state.allocation_pending
calls=[]
def fail(*args):raise AssertionError('worker-wide wait invoked')
worker=N(source_fence_jobs=set(),submit_store=lambda *a:calls.append(a) or True,wait=fail)
con=object.__new__(connector.GB10AlignedOffloadingConnector)
con.connector_worker=N(worker=worker,handle_preemptions=fail)
con._completion_enabled=False
meta=c.CompletionMetadata(load_jobs={},store_jobs={17:common.TransferJob('old','source','disk')},reserved_eviction_jobs={17})
con.prepare_completion(None,meta)
assert calls==[(17,'source','disk')] and not meta.store_jobs
assert not meta.reserved_eviction_jobs and not worker.source_fence_jobs
assert not meta.jobs_to_flush
print(json.dumps({'passed':True,'checks':['candidate modules import together',
    'slotted request state supports completion deferral','reserved copy submitted',
    'no handle_preemptions or wait invoked','metadata drained exactly once']}))
