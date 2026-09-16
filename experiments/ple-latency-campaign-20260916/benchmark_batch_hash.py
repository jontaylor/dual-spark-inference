"""Matched exact-row microbenchmark; separate temporary file, no serving changes."""
import ctypes as c,json,math,mmap,os,pathlib,random,statistics,tempfile,time
os.environ['OMP_WAIT_POLICY']='PASSIVE'
os.environ['OMP_DYNAMIC']='FALSE'
root=pathlib.Path('/home/jon/dual-spark-inference-kv-paging/results/ple-reader-investigation-20260916')
V=c.c_void_p;I=c.c_int64;U=c.c_uint64
libs={}
for name,path in [('direct-cache', '/home/jon/dual-spark-inference-kv-paging/experiments/ple-latency-campaign-20260916/variants/02-direct-cache/libple_batch_reader.so'), ('batch-hash', '/home/jon/dual-spark-inference-kv-paging/experiments/ple-latency-campaign-20260916/variants/10-batch-hash/libple_batch_reader.so')]:
 path=pathlib.Path(path)
 if not path.exists():continue
 lib=c.CDLL(str(path));lib.gb10_ple_reader_create.argtypes=[c.c_char_p,I,I,I,U,c.POINTER(c.c_int)];lib.gb10_ple_reader_create.restype=V
 lib.gb10_ple_reader_gather.argtypes=[V,V,I,V,c.c_uint];lib.gb10_ple_reader_destroy.argtypes=[V]
 libs[name]=lib
legacy=c.CDLL(str(root/'gather-legacy.so'));legacy.gb10_ple_gather.argtypes=[V,I,I,V,I,V,c.c_int]
rows=262144;width=160;rng=random.Random(119)
results=[]
with tempfile.TemporaryDirectory(prefix='ple-bench-') as td:
 path=pathlib.Path(td)/'table';data=rng.randbytes(rows*width);path.write_bytes(data)
 with path.open('r+b') as file:
  file.flush();os.fsync(file.fileno());mapped=mmap.mmap(file.fileno(),0,access=mmap.ACCESS_COPY);view=(c.c_char*len(data)).from_buffer(mapped)
  for count in (64,192,768,1024,1792,9000):
   ids_list=rng.sample(range(rows),count);ids=(I*count)(*ids_list);out=c.create_string_buffer(count*width)
   expected=b''.join(data[i*width:(i+1)*width] for i in ids_list)
   for cache in (0,16*2**20):
    handles={}
    for name,lib in libs.items():
     err=c.c_int();h=lib.gb10_ple_reader_create(os.fsencode(path),rows,width,131072,cache,c.byref(err));assert h,err.value;handles[name]=h
    calls={name:(lambda lib=libs[name],h=h:lib.gb10_ple_reader_gather(h,ids,count,out,10000)) for name,h in handles.items()}
    if cache==0:
     for threads in (1,16):calls[f'legacy-{threads}']=lambda threads=threads:legacy.gb10_ple_gather(view,rows,width,ids,count,out,threads)
    samples={name:[] for name in calls}
    for iteration in range(65):
     order=list(calls);rng.shuffle(order)
     for name in order:
      start=time.perf_counter_ns();ret=calls[name]();ns=time.perf_counter_ns()-start
      assert ret==0,(name,ret)
      if iteration<2:assert out.raw==expected,name
      if iteration>=5:samples[name].append(ns/1e6)
    for name,values in samples.items():
     result=dict(rows=count,cache_bytes=cache,variant=name,n=len(values),median_ms=statistics.median(values),p95_ms=sorted(values)[math.ceil(.95*len(values))-1]);results.append(result);print(json.dumps(result),flush=True)
    for name,h in handles.items():assert libs[name].gb10_ple_reader_destroy(h)==0
  del view;mapped.close()
(pathlib.Path('/home/jon/dual-spark-inference-kv-paging/results/ple-latency-campaign-20260916/batch-hash-benchmark.json')).write_text(json.dumps(results,indent=2)+'\n')
