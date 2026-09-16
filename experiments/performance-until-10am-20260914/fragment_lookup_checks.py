"""CPU integration: candidate lookup with real DiskSlotManager, no model/GPU."""
import json,runpy
from collections import OrderedDict
from types import SimpleNamespace as N
from vllm.v1.kv_offload.rank_local_disk import DiskSlotManager
from vllm.v1.kv_offload.base import ReqContext,LookupResult
mod=runpy.run_path('/tmp/fragment-cleanup-lookup-candidate.py')
Cache=mod['CompletionCache'];Record=mod['CompletionRecord'];digest=mod['prefix_digest']
m=DiskSlotManager(16);ctx=ReqContext('lookup-fixture')
keys=[b'missing',b'orphan',b'shared',b'valid',b'inherited']
m.prepare_store(keys,ctx);m.complete_store(keys,ctx);m.discard_idle_keys({b'missing'})
req=N(request_id='fixture',num_prompt_tokens=200,all_token_ids=[1]*200,cache_salt='lookup',mm_features=[],prompt_embeds=None,lora_request=None,skip_reading_prefix_cache=False)
state=N(transfer_jobs=[],req_context=ctx)
cache=Cache(N(connector_scheduler=N(manager=m,_req_status={'fixture':state})),[])
def rec(n,keys):return Record(n,digest(req.all_token_ids[:n],req.cache_salt),n,[(0,i,k) for i,k in enumerate(keys)])
bad=rec(128,[b'missing',b'orphan',b'shared',b'inherited']);good=rec(64,[b'shared',b'valid'])
cache.records=OrderedDict([((r.witness_length,r.digest),r) for r in [good,bad]])
cache.restored_pages={'other':{(0,0):b'inherited'}}
assert cache.lookup(req,0)==(64,True)
assert list(cache.records.values())==[good] and cache.selected['fixture'] is good
assert m.lookup(b'orphan',ctx)==LookupResult.MISS
assert all(m.lookup(k,ctx)==LookupResult.HIT for k in [b'shared',b'valid',b'inherited'])
assert state.num_locally_computed_tokens==0 and state.partial_tail_boundary is None
# Real lookup must preserve the selected shorter checkpoint on a repeat.
assert cache.lookup(req,0)==(64,True)
# An in-flight store remains pending rather than treated as a broken checkpoint.
pending=b'pending';m.prepare_store([pending],ctx);r=rec(128,[pending]);cache.records[(r.witness_length,r.digest)]=r
assert cache.lookup(req,0)==(None,False) and (r.witness_length,r.digest) in cache.records
m.complete_store([pending],ctx)
assert cache.lookup(req,0)==(128,True)
print(json.dumps({'passed':True,'actual_candidate_module_imported':True,'actual_disk_allocator':True,'broken_longer_falls_back_to_valid_shorter':True,'orphan_reclaimed':True,'shared_and_inherited_protected':True,'pending_then_ready_preserved':True,'scope':'CPU lookup/allocator integration only; no live serving deployment, model restore, or performance evidence.'},indent=2))
