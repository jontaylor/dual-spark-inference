"""Actual slot-manager pressure test; no GPU or model execution."""
import json
from vllm.v1.kv_offload.rank_local_disk import DiskSlotManager
from vllm.v1.kv_offload.base import ReqContext,LookupResult
ctx=ReqContext('fixture');shared=[b'shared-'+str(i).encode() for i in range(6)];m=DiskSlotManager(16);records=[]
def store(manager,keys):
 result=manager.prepare_store(keys,ctx);assert result is not None;assert result.keys_to_store==keys;manager.complete_store(keys,ctx);return result
store(m,shared)
for generation in range(5):
 m.prepare_load(shared,ctx)
 tails=[f'tail-{generation}-{i}'.encode() for i in range(2)]
 result=store(m,tails);assert not result.evicted_keys
 m.complete_load(shared,ctx);records.append(shared+tails)
assert all(m.lookup(k,ctx)==LookupResult.HIT for r in records for k in r)
assert m._get_num_free_blocks()==0 and m._num_evictable_cache_blocks==16
m.prepare_load(shared,ctx)
# An allocation larger than the unpinned capacity must fail without losing pins.
assert m.prepare_store([f'pressure-{i}'.encode() for i in range(11)],ctx) is None
assert all(m.lookup(k,ctx)==LookupResult.HIT for k in shared)
m.complete_load(shared,ctx)
assert m._num_evictable_cache_blocks==16
pressure=[f'pressure-{i}'.encode() for i in range(11)];store(m,pressure)
# Same16slots, fresh8-page copies for each checkpoint: older complete records lost.
baseline=DiskSlotManager(16);old_records=[]
for generation in range(5):
 keys=[f'copy-{generation}-{i}'.encode() for i in range(8)];store(baseline,keys);old_records.append(keys)
retained=sum(all(baseline.lookup(k,ctx)==LookupResult.HIT for k in r) for r in old_records)
assert retained==2
print(json.dumps({'scope':'Synthetic CPU allocator fixture, not live workload retention/speedup','slots':16,'checkpoints':5,'shared_full_pages':6,'fresh_tail_pages_per_checkpoint':2,'shared_complete_checkpoints_retained':5,'fresh_copy_complete_checkpoints_retained':retained,'pinned_overcapacity_rejected':True,'pins_released_before_later_eviction':True},indent=2))
