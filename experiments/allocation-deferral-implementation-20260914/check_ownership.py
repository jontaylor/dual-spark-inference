"""Exercise actual allocator/manager/copy-ACK code, without a model or CUDA."""
import importlib.util
import json
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace as N

P = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, P / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load('vllm.v1.kv_offload.gb10_allocation_deferral', 'deferral.py')
common = load('vllm.distributed.kv_transfer.kv_connector.v1.offloading.common', 'common.py')
bp = load('vllm.v1.core.block_pool', 'block_pool.py')
disk = load('vllm.v1.kv_offload.rank_local_disk', 'rank_local_disk.py')
pressure = load('vllm.v1.kv_offload.gb10_native_pressure', 'native_pressure.py')
completion = load('vllm.distributed.kv_transfer.kv_connector.v1.gb10_completion', 'completion.py')
from vllm.v1.kv_offload.gb10_allocation_deferral import AllocationDeferred
from vllm.v1.core.sched.parking_policy import ParkingPolicy
from vllm.v1.kv_offload.base import LookupResult, ReqContext, TransferResult


def fixture(size=12):
    pool = bp.BlockPool(size, True, 1920)
    ledger = ParkingPolicy(size - 1)
    manager = disk.DiskSlotManager(100)
    manager.configure_memory(pool, ledger)
    manager.native_mode = True
    ids = iter(range(100, 10000))
    scheduler = N(manager=manager, config=N(blocks_per_chunk=1, num_workers=2),
                  _generate_job_id=lambda: next(ids))
    connector = N(connector_scheduler=scheduler, _completion_groups=[1, 2, 3])
    cache = pressure.NativePressureCache(connector, pool)
    return pool, ledger, manager, cache


def publish(pool, manager, count=1):
    ctx = ReqContext('producer')
    blocks = pool.get_new_blocks(count)
    keys = [f'key-{b.block_id}'.encode() for b in blocks]
    spec = manager.prepare_memory_store(keys, ctx, {k: (0, i) for i, k in enumerate(keys)},
                                        borrowed={k: b.block_id for k, b in zip(keys, blocks)})
    manager.complete_store(keys, ctx)
    pool.free_blocks(blocks)
    return blocks, keys, ctx


def ack(cache, jobs, rank, final=False):
    cache.consume_completions(common.OffloadingWorkerMetadata(
        source_preserved={} if final else {j: {rank} for j in jobs},
        completed_jobs={j: 1 for j in jobs} if final else {}))


pool, ledger, manager, cache = fixture()
old, keys, ctx = publish(pool, manager, 3)
other = pool.get_new_blocks(8)
assert not cache.jobs
try:
    cache.ensure_available(3, request_id='A')
except AllocationDeferred:
    pass
else:
    raise AssertionError('A allocated before preservation')
assert pool.get_num_free_blocks() == 0
assert all(b.ref_cnt == 1 for b in old)
assert all(manager.lookup(k, ctx) == LookupResult.HIT_PENDING for k in keys)
# B can run without fresh allocation, while A waits. Source reservations persist.
cache.ensure_available(0, request_id='B')
assert all(b.ref_cnt == 1 for b in other)
meta = completion.CompletionMetadata(load_jobs={}, store_jobs={})
cache.add_metadata(meta)
jobs = set(cache.jobs)
assert meta.reserved_eviction_jobs == jobs and not meta.jobs_to_flush and not meta.native_eviction_jobs
ack(cache, jobs, 0)
ack(cache, jobs, 0)  # duplicate source ACK is idempotent
assert pool.get_num_free_blocks() == 0
ack(cache, jobs, 1)
assert pool.get_num_free_blocks() == 3
assert not manager.resident and not manager._resident_by_block
assert all(manager.lookup(k, ctx) == LookupResult.HIT_PENDING for k in keys)
reused = pool.get_new_blocks(3)
ack(cache, jobs, 1)  # late duplicate cannot free new owners
assert all(b.ref_cnt == 1 for b in reused)
ack(cache, jobs, 0, final=True)
assert all(manager.lookup(k, ctx) == LookupResult.HIT_PENDING for k in keys)
ack(cache, jobs, 1, final=True)
assert all(b.ref_cnt == 1 for b in reused) and not cache.jobs
assert all(manager.lookup(k, ctx) == LookupResult.HIT for k in keys)
assert not manager._detached_sources and not manager.spilling

# A free prefix-hit source is protected/adopted, not needlessly preserved.
p, l, m, c = fixture()
old, keys, ctx = publish(p, m)
other = p.get_new_blocks(10)
c.ensure_available(1, protected={old[0].block_id})
p.touch(old)
assert not c.jobs and old[0].ref_cnt == 1

# Bounded preservation even for a demand larger than the source reservation cap.
p, l, m, c = fixture(90)
old, keys, ctx = publish(p, m, 80)
other = p.get_new_blocks(9)
try:
    c.ensure_available(80)
except AllocationDeferred:
    pass
assert len(c.source_blocks) == 64
jobs = set(c.jobs)
ack(c, jobs, 0); ack(c, jobs, 1)
try:
    c.ensure_available(80)
except AllocationDeferred:
    pass
assert len(c.source_blocks) == 16

# A single physical page can back multiple storage keys. Both copies must ACK.
p, l, m, c = fixture(3)
old, keys, ctx = publish(p, m)
slot = next(iter(m.resident)); page = m.resident[slot]
p.touch(old)
r = m.prepare_memory_store([b'alias'], ctx, {b'alias': (1, 0)}, borrowed={b'alias': old[0].block_id})
m.complete_store([b'alias'], ctx); p.free_blocks(old)
other = p.get_new_blocks(1)
try: c.ensure_available(1)
except AllocationDeferred: pass
j1, j2 = c.jobs
ack(c, [j1], 0); ack(c, [j1], 1)
assert p.get_num_free_blocks() == 0
ack(c, [j2], 0); ack(c, [j2], 1)
assert p.get_num_free_blocks() == 1

# Completed persistence racing source notification must not lose the source ACK.
w = object.__new__(disk.RankLocalDiskWorker)
source, finished = Future(), Future()
w.jobs = {99: finished}; w.source_preserved = {99: source}; w.source_ack_ready = set()
finished.set_result(TransferResult(99, True, 1, .1))
assert w.get_finished() == []
source.set_result(None)
assert len(w.get_finished()) == 1
assert w.poll_source_preserved() == {99}
assert w.poll_source_preserved() == set()

# Source failures propagate instead of releasing a block.
w.source_preserved = {100: Future()}
w.source_preserved[100].set_exception(RuntimeError('injected copy failure'))
try: w.poll_source_preserved()
except RuntimeError: pass
else: raise AssertionError('copy error hidden')

a = completion.CompletionWorkerMetadata(source_preserved={1: {0}})
b = completion.CompletionWorkerMetadata(source_preserved={1: {1}})
assert a.aggregate(b).source_preserved == {1: {0, 1}}
print(json.dumps({'passed': True, 'checks': ['physical source reservation',
    'unrelated zero-allocation progress', 'no native worker fence', 'two-rank source ACK',
    'duplicate and late ACK', 'persistence gate', 'new owner survives final ACK',
    'protected prefix adoption', 'bounded multi-step reservation', 'multiple keys per block',
    'source/public completion race', 'copy failure propagation', 'metadata aggregation']}))
