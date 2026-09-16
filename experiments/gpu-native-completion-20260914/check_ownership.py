"""Actual allocator/slot-manager ownership gates; no model or GPU allocation."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as N

ROOT = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load('vllm.v1.core.block_pool', 'block_pool.gpu.py')
disk = load('vllm.v1.kv_offload.rank_local_disk', 'rank_local_disk.gpu.py')
pressure = load('vllm.v1.kv_offload.gb10_native_pressure', 'native_pressure.gpu.py')
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.parking_policy import ParkingPolicy
from vllm.v1.kv_offload.base import LookupResult, ReqContext
from vllm.distributed.kv_transfer.kv_connector.v1.gb10_completion import CompletionMetadata


def fixture():
    pool = BlockPool(12, True, 1920)
    ledger = ParkingPolicy(11)
    manager = disk.DiskSlotManager(20)
    manager.configure_memory(pool, ledger)
    manager.native_mode = True
    ids = iter(range(100, 1000))
    scheduler = N(manager=manager, config=N(blocks_per_chunk=1, num_workers=2),
                  _generate_job_id=lambda: next(ids))
    connector = N(connector_scheduler=scheduler, _completion_groups=[1, 2, 3])
    cache = pressure.NativePressureCache(connector, pool)
    return pool, ledger, manager, cache


pool, ledger, manager, cache = fixture()
ctx = ReqContext('producer')
source = pool.get_new_blocks(2)
keys = [b'attention', b'ring', b'recurrent']
locations = dict(zip(keys, [(0, 0), (1, 0), (2, 0)]))
result = manager.prepare_memory_store(keys, ctx, locations,
                                     borrowed=dict(zip(keys[:2], [b.block_id for b in source])))
assert result is not None and result.store_spec.direct_gpu
assert result.store_spec.memory_blocks[:2] == [b.block_id for b in source]
assert ledger.cache_reserved == 1  # Only new recurrent state, not attention history.
assert all(b.ref_cnt == 2 for b in source)
assert manager.lookup(keys[0], ctx) == LookupResult.HIT_PENDING
manager.complete_store(keys, ctx)
assert ledger.cache_reserved == 0
assert all(b.ref_cnt == 1 for b in source)
pool.free_blocks(source)
assert pool.get_num_free_blocks() == 11  # Completed pages are ordinary evictable GPU pages.
assert len(manager.resident) == 3 and not cache.jobs
assert manager.native_key(source[0].block_id, 0, 0, ctx) == keys[0]

# Native resident lookup touches the real allocator so a load cannot race reuse.
spec = manager.prepare_load(keys, ctx)
assert spec.direct_gpu and spec.memory_blocks == result.store_spec.memory_blocks
assert pool.get_num_free_blocks() == 8
assert all(manager.resident[s].load_pins == 1 for s in spec.block_ids.tolist())
manager.complete_load(keys, ctx)
assert pool.get_num_free_blocks() == 11 and ledger.cache_reserved == 0

# All previously untouched blocks can be allocated with no disk traffic.
cache.add_metadata(CompletionMetadata(load_jobs={}, store_jobs={}))
plain = pool.get_new_blocks(8)
assert not cache.jobs and pool.get_num_free_blocks() == 3
# Only selecting the cached physical pages triggers their backing writes.
reused = pool.get_new_blocks(3)
assert len(cache.jobs) == 3
assert all(manager.lookup(k, ctx) == LookupResult.HIT_PENDING for k in keys)
assert all(b.ref_cnt == 1 for b in reused)
meta = CompletionMetadata(load_jobs={}, store_jobs={})
cache.add_metadata(meta)
assert set(meta.store_jobs) == meta.jobs_to_flush == meta.native_eviction_jobs
first = N(completed_jobs={jid: 1 for jid in cache.jobs})
cache.consume_completions(first)
assert len(cache.jobs) == 3 and len(manager.resident) == 3
cache.consume_completions(first)
assert not cache.jobs and not manager.resident and not manager._resident_by_block
# Spill completion must never free the physical block now owned by another request.
assert all(b.ref_cnt == 1 for b in reused)
assert pool.get_num_free_blocks() == 0 and ledger.cache_reserved == 0
assert all(manager.lookup(k, ctx) == LookupResult.HIT for k in keys)
disk_spec = manager.prepare_load(keys, ctx)
assert not isinstance(disk_spec, disk.ResidentSlots)
manager.complete_load(keys, ctx)
pool.free_blocks(plain + reused)

# Retiring metadata or resetting must not decrement unrelated allocator owners.
p, l, m, c = fixture()
s = p.get_new_blocks(1)[0]
r = m.prepare_memory_store([b'a'], ctx, {b'a': (0, 0)}, borrowed={b'a': s.block_id})
m.complete_store([b'a'], ctx)
assert s.ref_cnt == 1
m.discard_idle_keys([b'a'])
assert s.ref_cnt == 1 and not m.resident and l.cache_reserved == 0
p.free_blocks([s])
r = m.prepare_memory_store([b'new'], ctx, {b'new': (2, 0)})
assert l.cache_reserved == 1
m.complete_store([b'new'], ctx, success=False)
assert l.cache_reserved == 0 and not m.resident and p.get_num_free_blocks() == 11
m.reset_cache()
print(json.dumps({'passed': True, 'new_allocations_only_for_recurrent_state': True,
                  'completed_pages_return_to_native_lru': True, 'no_permanent_reservation': True,
                  'loads_pin_physical_sources': True, 'writes_only_on_actual_reuse': True,
                  'both_rank_ack_gate': True, 'reuse_owner_not_freed_on_spill_ack': True,
                  'failure_and_retirement_release_exact_owners': True}), flush=True)
