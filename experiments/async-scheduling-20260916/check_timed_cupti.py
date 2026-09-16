"""Disposable concurrent CUDA work; tracer detaches at a driver API exit."""
import ctypes,json,time,threading
from pathlib import Path
import torch
start=time.time()
a=torch.zeros(1024,device='cuda');b=torch.ones_like(a)
stream=torch.cuda.Stream()
with torch.cuda.stream(stream):a.add_(b)
torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):a.add_(b)
g.replay();torch.cuda.synchronize()
base=a[0].item()
lib=ctypes.CDLL('/tmp/async-timed-trace.so')
lib.trace_start_timed.argtypes=[ctypes.c_char_p,ctypes.c_uint]
assert lib.trace_start_timed(b'/tmp/async-timed-cupti.jsonl',1500)==0
stop=threading.Event();errors=[]
def background():
 try:
  s=torch.cuda.Stream();x=torch.zeros(1024,device='cuda')
  with torch.cuda.stream(s):
   while not stop.is_set():
    x.add_(1);s.synchronize();time.sleep(.003)
 except Exception as error:errors.append(repr(error))
thread=threading.Thread(target=background);thread.start()
replays=0
while time.time()-start<20:
 g.replay();replays+=1
 torch.cuda.synchronize()
 if lib.trace_status() in (-1,4):break
 time.sleep(.002)
stop.set();thread.join(timeout=5)
assert not thread.is_alive() and not errors,errors
assert lib.trace_status()==4,lib.trace_status()
for _ in range(1000):g.replay();replays+=1
torch.cuda.synchronize()
assert a[0].item()==base+replays
rows=[json.loads(l) for l in Path('/tmp/async-timed-cupti.jsonl').read_text().splitlines()]
k=[r for r in rows if r.get('kind')=='kernel' and r['end']>r['start']>0]
assert len(k)>10
assert any(r.get('event')=='collection_disabled' and r.get('result')==0 for r in rows) and rows[-1]['result']==0 and rows[-1]['dropped']==0
print(json.dumps(dict(passed=True,start_wall=start,end_wall=time.time(),replays=replays,kernels=len(k),post_disable_graph_valid=True)))
