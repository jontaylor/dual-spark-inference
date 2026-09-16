"""A successful lookup must survive allocation, without pinning waiting requests."""
import json
import runpy
from pathlib import Path
from types import SimpleNamespace as N

scope=runpy.run_path(str(Path(__file__).with_name('check_finish_plan.py')))
pool,manager,cache,s,c=[scope[k] for k in ('pool','manager','cache','s','c')]
ReqContext=scope['ReqContext']
rid='follow-up';ctx=ReqContext(rid)
s._req_status[rid]=N(req_context=ctx,transfer_jobs=set())
request=N(request_id=rid,all_token_ids=list(range(4300)),cache_salt=None,num_prompt_tokens=4300,
          mm_features=[],prompt_embeds=None,lora_request=None,skip_reading_prefix_cache=False)
hit=cache.lookup(request,3840)
assert hit==(270,True)
keys,context=cache.lookup_pins[rid]
assert len(keys)==2  # one attention tail and one recurrent state; full local pages need no pin
pages=[manager.resident[manager._policy.get(k).block_id] for k in keys]
assert all(p.load_pins==1 and p.block.ref_cnt>0 for p in pages)
selected_ids={p.block.block_id for p in pages}
allocated=pool.get_new_blocks(pool.get_num_free_blocks())
assert not selected_ids.intersection(b.block_id for b in allocated)
# Successful allocation hands lifetime from the lookup to the actual load.
manager.prepare_load(keys,context)
cache.release_lookup_pin(rid)
assert all(p.load_pins==1 and p.block.ref_cnt>0 for p in pages)
manager.complete_load(keys,context)
assert all(p.load_pins==0 for p in pages)
pool.free_blocks(allocated)
# An allocation failure / skipped waiter must release the provisional pin.
assert cache.lookup(request,3840)==(270,True)
cache.add_metadata(c.CompletionMetadata(load_jobs={},store_jobs={}))
assert not cache.lookup_pins and all(p.load_pins==0 for p in pages)
print(json.dumps({'passed':True,'lookup_sources_survive_destination_allocation':True,
                  'load_handoff_preserves_pin':True,'unscheduled_waiter_releases_pins':True}),flush=True)
