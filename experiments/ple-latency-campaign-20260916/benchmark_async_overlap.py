"""Compare submission cost and completion for inline versus forced async issue.

Dedicated temporary file only. Projected GPU delay assumes ideal independent
GPU work and excludes graph-launch overhead; it is not a live serving result.
"""
import argparse
import ctypes as c
import json
import os
from pathlib import Path
import random
import statistics
import tempfile
import time

base=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--variant',default='08-async-overlap');args=parser.parse_args()
lib=c.CDLL(str(base/'variants'/args.variant/'libple_batch_reader.so'))
v,i=c.c_void_p,c.c_int64
lib.gb10_ple_reader_create.argtypes=[c.c_char_p,i,i,i,c.c_uint64,c.POINTER(c.c_int)]
lib.gb10_ple_reader_create.restype=v
for name in ['submit','submit_async']:
 getattr(lib,'gb10_ple_reader_'+name).argtypes=[v,v,i,v,c.c_uint]
lib.gb10_ple_reader_complete.argtypes=[v]
lib.gb10_ple_reader_destroy.argtypes=[v]
rng=random.Random(917);rows=262144;width=160;results=[]
with tempfile.TemporaryDirectory(prefix='ple-async-overlap-') as td:
 path=Path(td)/'table';data=rng.randbytes(rows*width);path.write_bytes(data)
 fd=os.open(path,os.O_RDONLY);os.fsync(fd)
 for cold in [False,True]:
  for count in [64,192,768,1536]:
   error=c.c_int();handles={name:lib.gb10_ple_reader_create(os.fsencode(path),rows,width,count,0,c.byref(error)) for name in ['submit','submit_async']}
   assert all(handles.values()),error.value
   samples={name:[] for name in handles}
   for iteration in range(45):
    ids_list=rng.sample(range(rows),count);ids=(i*count)(*ids_list);out=c.create_string_buffer(count*width)
    expected=b''.join(data[x*width:(x+1)*width] for x in ids_list)
    order=list(handles);rng.shuffle(order)
    for name in order:
     if cold:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
     a=time.perf_counter_ns();rc=getattr(lib,'gb10_ple_reader_'+name)(handles[name],ids,count,out,10000);b=time.perf_counter_ns()
     assert rc==0,rc
     assert lib.gb10_ple_reader_complete(handles[name])==0
     d=time.perf_counter_ns();assert out.raw==expected
     if iteration>=5:samples[name].append({'submit_ms':(b-a)/1e6,'complete_ms':(d-b)/1e6,'total_ms':(d-a)/1e6})
   for name,sample in samples.items():
    result={'cold_advised':cold,'rows':count,'variant':name,'n':len(sample)}
    for key in ['submit_ms','complete_ms','total_ms']:
     values=sorted(x[key] for x in sample);result[key]={'median':statistics.median(values),'p95':values[int(.95*len(values))]}
    for early in [.5,1,2,3]:
     values=sorted(x['submit_ms']+max(0,x['complete_ms']-early) for x in sample)
     result[f'ideal_exposed_ms_with_{early}ms_early_gpu']={'median':statistics.median(values),'p95':values[int(.95*len(values))]}
    results.append(result);print(json.dumps(result),flush=True)
   for handle in handles.values():assert lib.gb10_ple_reader_destroy(handle)==0
 os.close(fd)
(base/'variants'/args.variant/'benchmark.json').write_text(json.dumps(results,indent=2)+'\n')
