"""Bounded GPU microbenchmark of the nonterminal marker/copy fast path.

Runs alongside serving with isolated tensors: contention affects these timings;
this is added candidate cost, not a sync-versus-async throughput comparison.
"""
import importlib.util,json,statistics,sys,time
import torch
spec=importlib.util.spec_from_file_location('candidate_terminal',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
mark=m.make_marker();copy=m.make_snapshot_copier()
start_wall=time.time();results=[]
for n in (12,32):
 device='cuda'
 mapping=torch.arange(n,dtype=torch.int32,device=device)
 sampled=torch.full((n,4),8,dtype=torch.int64,device=device)
 counts=torch.full((n,),4,dtype=torch.int32,device=device)
 after=torch.full((n,),104,dtype=torch.int32,device=device)
 prompts=torch.full((n,),50,dtype=torch.int32,device=device)
 limits=torch.full((n,),1000,dtype=torch.int32,device=device)
 eos=torch.full((n,),-1,dtype=torch.int32,device=device)
 stops=torch.full((n,1),-1,dtype=torch.int64,device=device)
 stopcounts=torch.zeros_like(eos);boundaries=torch.full_like(eos,-1);steps=torch.full_like(eos,-1)
 descriptors=torch.zeros((n*86,15),dtype=torch.int64,device=device)
 descriptors[:,7]=torch.arange(n,device=device).repeat_interleave(86)
 columns=torch.zeros_like(eos)
 def launch():
  mark[(n,)](mapping,sampled,counts,after,prompts,limits,eos,stops,stopcounts,boundaries,steps,4,1,262144,1,STOP_TILE=1)
  copy[(4,n*86)](descriptors,after,counts,columns,boundaries,steps,1,MAX_SPEC=3,TILE=4096,COPY_LANES=4)
 launch();torch.cuda.synchronize()
 times=[]
 for _ in range(30):
  begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
  begin.record();launch();end.record();end.synchronize()
  times.append(begin.elapsed_time(end)*1000)
 assert bool(steps.eq(-1).all())
 results.append(dict(requests=n,pieces_per_request=86,samples=30,gpu_elapsed_us=times,
  median_us=statistics.median(times),p95_us=sorted(times)[28],max_us=max(times)))
print(json.dumps(dict(scope='isolated nonterminal candidate work alongside live serving; contention included; not a throughput comparison',
 start_wall=start_wall,end_wall=time.time(),results=results,peak_allocated_bytes=torch.cuda.max_memory_allocated()),indent=2))
