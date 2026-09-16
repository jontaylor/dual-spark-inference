"""Real-size GPU pages, source ACK before held persistence, exact disk restore."""
import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
import torch
P=Path(__file__).parent
def load(name,file):
    spec=importlib.util.spec_from_file_location(name,P/file)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
m=load('deferral_disk_test','rank_local_disk.py')
from vllm.v1.kv_offload.base import CanonicalKVCaches,CanonicalKVCacheTensor,CanonicalKVCacheRef,GPULoadStoreSpec
rank=int(sys.argv[1]);torch.cuda.set_device(0)
size=27156480
tensor=torch.empty((3,size),device='cuda',dtype=torch.uint8);tensor[0].fill_(71)
layout=CanonicalKVCaches([CanonicalKVCacheTensor(tensor,size)],[[CanonicalKVCacheRef(0,size)]])
w=m.RankLocalDiskWorker(layout,'/tmp/deferral-gpu-test',str(time.time_ns()),rank,2,size,True)
gate=threading.Event();entered=threading.Event();real=w.content_pages.store
try:
    def delayed(*args):
        entered.set()
        if not gate.wait(20):raise TimeoutError('held persistence timed out')
        return real(*args)
    w.content_pages.store=delayed;w.source_fence_jobs={101}
    w.submit_store(101,GPULoadStoreSpec([0],[1],[0]),m.DiskSlots([0]))
    w.source_fence_jobs.clear()
    # Submission returns without invoking wait(); unrelated GPU work can run.
    tensor[2].fill_(13);torch.cuda.synchronize()
    assert bool((tensor[2]==13).all()) and not w.jobs[101].done()
    end=time.monotonic()+10
    while True:
        acks=w.poll_source_preserved()
        if 101 in acks:break
        assert time.monotonic()<end
        time.sleep(.001)
    assert entered.wait(3) and not w.jobs[101].done() and w.get_finished()==[]
    tensor[0].fill_(99);torch.cuda.synchronize()
    gate.set();w.wait({101});assert w.get_finished()[0].success
    w.submit_load(102,m.DiskSlots([0]),GPULoadStoreSpec([1],[1],[0]))
    w.wait({102});assert w.get_finished()[0].success
    assert bool((tensor[1]==71).all())
    print(json.dumps({'passed':True,'rank':rank,'page_bytes':size,
        'checks':['nonblocking submission','unrelated GPU work progresses with held disk write',
                  'source ACK before final persistence','overwrite original then exact disk restore',
                  'checksum and GPU readback retained']}))
finally:
    gate.set();w.shutdown()
