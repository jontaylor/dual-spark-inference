"""Exercise real allocator/slot ownership with no inference or GPU allocation."""
import ast
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as N

from vllm.v1.core.kv_cache_utils import make_block_hash_with_group_id
from vllm.v1.kv_offload.base import LookupResult
from vllm.v1.kv_offload.rank_local_disk import DiskSlotManager


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, '/tmp/' + filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bp = load('native_test_block_pool', 'block_pool.native.py')
np = load('native_test_pressure', 'native_pressure.py')
cm = load('native_test_completion', 'completion.native.py')
connector_module = load('native_test_connector', 'connector.native.py')


def fixture(slots=16):
    pool = bp.BlockPool(8, True, 64)
    manager = DiskSlotManager(slots)
    counter = iter(range(100, 1000))
    scheduler = N(manager=manager, config=N(blocks_per_chunk=1, num_workers=2,
        kv_group_configs=[N(tokens_per_chunk=1920, group_idx=0)]),
        _generate_job_id=lambda: next(counter))
    connector = N(connector_scheduler=scheduler, _index={3:0}, _completion_groups=[1,2])
    cache = np.NativePressureCache(connector, pool)
    return pool, manager, cache


def publish(pool, block, value, boundary=1920, group=3):
    pool._insert_block_hash(make_block_hash_with_group_id(bytes([value])*32, group), block, boundary)


def meta():
    return cm.CompletionMetadata(load_jobs={}, store_jobs={})


pool, manager, cache = fixture()
blocks = pool.get_new_blocks(2)
publish(pool, blocks[0], 1)
publish(pool, blocks[1], 2)
pool.free_blocks(blocks)
cache.add_metadata(meta())
assert pool.get_num_free_blocks() == 7
assert not manager.resident and not cache.jobs
unused = pool.get_new_blocks(5)
assert not cache.jobs  # No write while untouched physical blocks remain.
reused = pool.get_new_blocks(1)[0]
assert reused is blocks[0] and reused.ref_cnt == 1
assert len(cache.jobs) == 1 and not manager.resident
assert pool.get_num_free_blocks() == 1  # No duplicate cache allocation/pinning.
m = meta(); cache.add_metadata(m)
jid = next(iter(m.store_jobs)); job = m.store_jobs[jid]
key = next(iter(cache.jobs[jid].keys))
assert m.jobs_to_flush == m.native_eviction_jobs == {jid}
assert job.src_spec.block_ids.tolist() == [reused.block_id]
assert job.src_spec.group_sizes == [1, 0]
assert manager.lookup(key, cache.context) == LookupResult.HIT_PENDING
assert cache.consume_completions(N(completed_jobs={jid:1, 999:1})) == {999:1}
assert manager.lookup(key, cache.context) == LookupResult.HIT_PENDING
assert cache.consume_completions(N(completed_jobs={jid:1})) == {}
assert manager.lookup(key, cache.context) == LookupResult.HIT
assert not cache.jobs

# The next reuse of the same backed content must not write again.
publish(pool, reused, 1)
pool.free_blocks([reused])
cache.add_metadata(meta())
other = pool.get_new_blocks(1)[0]
assert other is blocks[1]
assert len(cache.jobs) == 1
again = pool.get_new_blocks(1)[0]
assert again is reused and len(cache.jobs) == 1

# A newly scheduled hash does not represent bytes produced by the GPU yet.
pool2, _, cache2 = fixture()
b = pool2.get_new_blocks(1)[0]
publish(pool2, b, 9)
pool2.free_blocks([b])
pool2.get_new_blocks(7)
assert not cache2.jobs

# Partial and omitted ring entries cannot be published as aligned disk hits.
for boundary, group in [(64, 3), (1920, 7)]:
    p, _, c = fixture()
    b = p.get_new_blocks(1)[0]
    publish(p, b, 5, boundary, group)
    p.free_blocks([b]); c.add_metadata(meta()); p.get_new_blocks(7)
    assert not c.jobs

# Explicit invalidation is not a request to back potentially invalid data.
p, _, c = fixture(); b = p.get_new_blocks(1)[0]
publish(p, b, 6); p.free_blocks([b]); c.add_metadata(meta())
p.evict_blocks({b.block_id}); assert not c.jobs

# Exercise the actual early worker fence using the production submission loop.
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import OffloadingConnectorWorker
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingWorkerMetadata
memory = {reused.block_id: b'old-native-prefix'}
stored = {}
class Transport:
    def submit_store(self, jid, src, dst):
        stored[jid] = memory[int(src.block_ids[0])]
        return True
    def wait(self, jobs):
        assert all(j in stored for j in jobs)
worker = OffloadingConnectorWorker.__new__(OffloadingConnectorWorker)
worker.worker = Transport()
worker._unsubmitted_store_jobs = []
worker._is_store_writer = True
worker._connector_worker_meta = OffloadingWorkerMetadata()
connector = connector_module.GB10AlignedOffloadingConnector.__new__(connector_module.GB10AlignedOffloadingConnector)
connector.connector_worker = worker
connector._completion_enabled = False
connector.prepare_completion(N(), m)
memory[reused.block_id] = b'new-request-overwrite'
assert stored[jid] == b'old-native-prefix'
assert not m.store_jobs and not m.jobs_to_flush and not m.native_eviction_jobs

# Ordinary completion must not invoke snapshot capture in pressure-only mode.
from unittest.mock import patch
connector.pressure_only = True; connector.memory_completions = False
connector._pressure_saves = {}
connector._completion = N(cancel_active=lambda rid:False, restored_pages={},
    finish=lambda *a, **k: (_ for _ in ()).throw(AssertionError('proactive capture')))
with patch.object(connector_module.OffloadingConnector, 'request_finished_all_groups', return_value=(False,None)):
    assert connector.request_finished_all_groups(N(request_id='done'), ([1],)) == (False,None)

runner = ast.parse(Path('/tmp/model_runner.deep.py').read_text())
fn = next(n for n in ast.walk(runner) if isinstance(n, ast.FunctionDef) and n.name=='execute_model')
calls = [(n.lineno, ast.unparse(n.func)) for n in ast.walk(fn) if isinstance(n,ast.Call)]
early = min(line for line,name in calls if name=='completion_hook')
for method in ('finish_requests','free_states','add_requests','update_requests'):
    assert early < min(line for line,name in calls if name=='self.'+method)
print(json.dumps({'passed':True, 'gates':['no-pressure zero stores','zero duplicate GPU blocks',
    'both rank ACKs before publication','backed-page no rewrite','same-step stale hash rejection',
    'partial/ring rejection','invalidation does not persist','pre-overwrite worker fence',
    'ordinary completion no snapshot','fence precedes request mutation']}))
