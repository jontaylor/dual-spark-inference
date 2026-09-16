"""Disposable process only: attach activity collection after CUDA graph creation."""
import ctypes,json,time,sys
from pathlib import Path
import torch
start=time.time()
early=len(sys.argv)>1 and sys.argv[1]=="early"
if early:
 lib=ctypes.CDLL('/tmp/async-live-trace-probe.so')
 lib.trace_begin.argtypes=[ctypes.c_char_p];lib.trace_begin.restype=ctypes.c_int
 assert lib.trace_begin(b'/tmp/async-late-cupti.jsonl')==0
a=torch.ones((256,256),device='cuda');b=torch.ones_like(a)
stream=torch.cuda.Stream()
with torch.cuda.stream(stream):
 for _ in range(3):a.add_(b)
torch.cuda.synchronize()
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):a.add_(b)
graph.replay();torch.cuda.synchronize()
if not early:
 lib=ctypes.CDLL('/tmp/async-live-trace-probe.so')
 lib.trace_begin.argtypes=[ctypes.c_char_p];lib.trace_begin.restype=ctypes.c_int
 assert lib.trace_begin(b'/tmp/async-late-cupti.jsonl')==0
for _ in range(20):graph.replay()
a.mul_(b)
torch.cuda.synchronize()
assert lib.trace_stop_after_sync()==0
# Verify application and existing graph still work after CUPTI detaches.
before=a[0,0].item();graph.replay();torch.cuda.synchronize()
assert a[0,0].item()==before+1
rows=[json.loads(line) for line in Path('/tmp/async-late-cupti.jsonl').read_text().splitlines()]
kernels=[r for r in rows if r.get('kind')=='kernel' and r['end']>r['start']>0]
assert len(kernels)>=21, f'Only {len(kernels)} timed kernel records; tracing did not work'
print(json.dumps(dict(passed=True,early=early,kernel_records=len(kernels),start_wall=start,end_wall=time.time(),post_detach_graph_valid=True)))
