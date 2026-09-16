"""Matched cold/warm file reads; cache eviction affects only a temporary file."""
import ctypes as c,json,math,mmap,os,pathlib,random,statistics,tempfile,time
root=pathlib.Path(__file__).resolve().parent;out=root.parents[1]/'results/ple-latency-campaign-20260916';rng=random.Random(77)
V=c.c_void_p;I=c.c_int64;rows=262144;width=160;data=rng.randbytes(rows*width)
libs={}
for name,variant in [('buffered','02-direct-cache'),('direct','04-direct-io'),('hybrid','05-hybrid-io')]:
 lib=c.CDLL(str(root/'variants'/variant/'libple_batch_reader.so'));lib.gb10_ple_reader_create.argtypes=[c.c_char_p,I,I,I,c.c_uint64,c.POINTER(c.c_int)];lib.gb10_ple_reader_create.restype=V;lib.gb10_ple_reader_gather.argtypes=[V,V,I,V,c.c_uint];lib.gb10_ple_reader_destroy.argtypes=[V];libs[name]=lib
results=[]
with tempfile.TemporaryDirectory(prefix='ple-direct-cold-') as td:
 path=pathlib.Path(td)/'table';path.write_bytes(data)
 with path.open('rb') as file:
  os.fsync(file.fileno())
  for count in (64,192,768,1792):
   ids_list=rng.sample(range(rows),count);ids=(I*count)(*ids_list);output=c.create_string_buffer(count*width);expected=b''.join(data[x*width:(x+1)*width] for x in ids_list)
   handles={}
   for name,lib in libs.items():
    err=c.c_int();h=lib.gb10_ple_reader_create(os.fsencode(path),rows,width,10000,0,c.byref(err));assert h,err.value;handles[name]=h
   for mode in ('cold-advised','warm-file'):
    samples={name:[] for name in libs}
    for iteration in range(25):
     order=list(libs);rng.shuffle(order)
     for name in order:
      if mode=='cold-advised':os.posix_fadvise(file.fileno(),0,0,os.POSIX_FADV_DONTNEED)
      else:
       for offset in ids_list:assert len(os.pread(file.fileno(),width,offset*width))==width
      start=time.perf_counter_ns();rc=libs[name].gb10_ple_reader_gather(handles[name],ids,count,output,10000);elapsed=(time.perf_counter_ns()-start)/1e6
      assert rc==0,(name,rc);assert output.raw==expected
      if iteration>=5:samples[name].append(elapsed)
    for name,a in samples.items():
     result=dict(variant=name,mode=mode,rows=count,n=len(a),median_ms=statistics.median(a),p95_ms=sorted(a)[math.ceil(.95*len(a))-1]);results.append(result);print(json.dumps(result),flush=True)
   for name,h in handles.items():assert libs[name].gb10_ple_reader_destroy(h)==0
(out/'hybrid-io-benchmark.json').write_text(json.dumps(results,indent=2)+'\n')
