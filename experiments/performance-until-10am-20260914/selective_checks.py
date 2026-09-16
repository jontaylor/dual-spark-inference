import importlib.util,sys,types,itertools,json
from unittest.mock import patch
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec,MambaSpec,CircularBufferSpec
from vllm.v1.kv_offload.base import GPULoadStoreSpec,LookupResult
N=types.SimpleNamespace
sp=importlib.util.spec_from_file_location('selective_under_test','/tmp/completion.selective.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
attention=FullAttentionSpec(block_size=64,num_kv_heads=1,head_size=2,dtype=torch.bfloat16)
recurrent=MambaSpec(64,((2,6),(3,2,2)),(torch.float32,torch.float32))
ring=CircularBufferSpec(8,num_kv_heads=1,head_size=2,dtype=torch.uint8)
groups=[N(kv_cache_spec=x) for x in [attention,recurrent,ring]]
class Manager:
 def __init__(self,missing=(),fail=False):self.missing=set(missing);self.fail=fail;self.pins=[];self.released=[];self.new=[]
 def lookup(self,k,c):return LookupResult.MISS if k in self.missing else LookupResult.HIT
 def prepare_load(self,keys,c):self.pins.append(list(keys));return GPULoadStoreSpec([1]*len(keys),[len(keys)],[0])
 def complete_load(self,keys,c):self.released+=list(keys)
 def prepare_memory_store(self,keys,c,locations):return self.prepare_store(keys,c)
 def prepare_store(self,keys,c):
  self.new=list(keys)
  return None if self.fail else N(keys_to_store=list(keys),store_spec=N())
def run(missing=(),fail=False,eagle=False,active=False):
 ids=itertools.count(1);manager=Manager(missing,fail)
 state=N(req_context=N(req_id='r'),num_locally_computed_tokens=0,transfer_jobs=set(),group_states=[N(block_ids=[],offload_keys=[]) for _ in groups],update_offload_keys=lambda:None)
 s=N(manager=manager,_req_status={'r':state},config=N(kv_group_configs=[N(is_eagle_group=eagle,sliding_window_size_in_chunks=None) for _ in groups],num_workers=2),_generate_job_id=lambda:next(ids),_jobs={},_current_batch_load_jobs={})
 cache=m.CompletionCache(N(connector_scheduler=s,_alignment=64),groups)
 req=N(request_id='r',num_prompt_tokens=180,num_computed_tokens=192,num_tokens=193,all_token_ids=list(range(193)),cache_salt='salt',status=m.RequestStatus.FINISHED_LENGTH_CAPPED,num_in_flight_tokens=0,mm_features=[],prompt_embeds=None,lora_request=None,skip_reading_prefix_cache=False)
 record=m.CompletionRecord(161,b'digest',160,[(0,0,b'full-0'),(0,1,b'local-1'),(0,2,b'partial-2'),(1,2,b'recurrent'),(2,0,b'ring')]);cache.selected['r']=record
 blocks=N(blocks=[[N(block_id=10+i,is_null=False,block_hash=b'local' if i==1 else None) for i in range(3)],[N(block_id=20+i,is_null=False,block_hash=None) for i in range(3)],[N(block_id=30,is_null=False,block_hash=None)]])
 assert cache.allocate(req,blocks,160)
 # Only full-0 was actually loaded and lies wholly before restored boundary.
 assert cache.restored_pages['r']=={(0,0):b'full-0'}
 state.transfer_jobs.clear();manager.pins.clear()
 ok=cache.finish(req,([10,11,12],[20,21,22],[30]),memory=True,active=active)
 if fail:
  assert not ok and manager.released==[b'full-0'];return
 assert ok
 save,shared,_=cache.pending['r'];expected=[] if b'full-0' in missing else [b'full-0']
 assert shared==expected
 assert len(manager.new)==5-len(expected)
 assert all(k not in [b'local-1',b'partial-2',b'recurrent',b'ring'] for _,_,k in save.record.pages)
 assert (('r' in cache.restored_pages)==active)
 # Shared dependencies pinned before allocation. Worker sees only new sources.
 assert manager.pins==([expected] if expected else [])
 assert sum(cache.to_send['r'][1].src_spec.group_sizes)==len(manager.new)
for kw in [{},{'missing':[b'full-0']},{'fail':True},{'eagle':True},{'active':True}]:run(**kw)
print(json.dumps({'actual_restored_full_page_reused':True,'local_shared_page_excluded':True,'partial_tail_excluded':True,'recurrent_and_ring_excluded':True,'evicted_key_falls_back_to_new_copy':True,'allocation_failure_releases_shared_pins':True,'active_capture_retains_provenance':True,'finished_capture_cleans_provenance':True,'eagle_boundary_conservative':True},indent=2))
