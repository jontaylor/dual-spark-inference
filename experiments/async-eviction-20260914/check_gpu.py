import importlib.util,sys,time,threading,json
from pathlib import Path
import torch
p=Path(__file__).parent
def load(name,file):
 spec=importlib.util.spec_from_file_location(name,p/file);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
load('vllm.v1.kv_offload.gb10_staged_evictions','pipeline.py')
m=load('async_disk_candidate','rank_local_disk.async.py')
from vllm.v1.kv_offload.base import CanonicalKVCaches,CanonicalKVCacheTensor,CanonicalKVCacheRef,GPULoadStoreSpec
rank=int(sys.argv[1]);torch.cuda.set_device(0)
# Actual physical page byte size; deliberately block persistence while GPU is overwritten.
size=27156480
tensor=torch.arange(4*size,device='cuda',dtype=torch.int32).remainder(251).to(torch.uint8).reshape(4,size)
layout=CanonicalKVCaches([CanonicalKVCacheTensor(tensor,size)],[[CanonicalKVCacheRef(0,size)]])
w=m.RankLocalDiskWorker(layout,'/tmp/async-eviction-test',str(time.time_ns()),rank,2,size,True)
gate=threading.Event();entered=threading.Event();real=w.content_pages.store
try:
 def delayed(*args):entered.set();assert gate.wait(10);return real(*args)
 w.content_pages.store=delayed
 w.source_fence_jobs={101}
 expected=tensor[0].clone()
 w.submit_store(101,GPULoadStoreSpec([0],[1],[0]),m.DiskSlots([0]))
 w.wait({101});assert entered.wait(3)
 assert not w.jobs[101].done();assert w.get_finished()==[]
 tensor[0].fill_(77);torch.cuda.synchronize()
 w.source_fence_jobs.clear();gate.set();w.wait({101});assert w.get_finished()[0].success
 w.submit_load(102,m.DiskSlots([0]),GPULoadStoreSpec([1],[1],[0]));w.wait({102});assert w.get_finished()[0].success
 assert torch.equal(tensor[1],expected),'overwrite corrupted staged restore'
 # More jobs than staging slots, alternating source generations and disk slots.
 w.content_pages.store=real
 for jid in range(110,130):
  tensor[0].fill_(jid);w.source_fence_jobs={jid}
  w.submit_store(jid,GPULoadStoreSpec([0],[1],[0]),m.DiskSlots([jid]));w.wait({jid});w.source_fence_jobs.clear()
 w.wait(set(range(110,130)));assert len(w.get_finished())==20
 for jid in range(110,130):
  w.submit_load(jid+100,m.DiskSlots([jid]),GPULoadStoreSpec([2],[1],[0]));w.wait({jid+100});w.get_finished();assert bool((tensor[2]==jid).all())
 print(json.dumps({'passed':True,'rank':rank,'page_bytes':size,'staging_bytes':len(w.eviction_view),'checks':['source fence before disk ACK','overwrite then exact checked disk restore','20 generations across8 buffers','checksum and GPU readback']}))
finally:
 gate.set();w.shutdown()
