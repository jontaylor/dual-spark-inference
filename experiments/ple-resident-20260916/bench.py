import ctypes as c,json,os,pathlib,random,statistics,tempfile,time,mmap
os.environ['OMP_WAIT_POLICY']='PASSIVE'
P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]/'results/ple-resident-20260916'; V=c.c_void_p;I=c.c_int64
libs={}
for name,path in [('baseline',P.parent/'ple-latency-campaign-20260916/variants/17-fixed-policy/libple_batch_reader.so'),('resident',P/'libple_batch_reader.so')]:
 lib=c.CDLL(str(path));lib.gb10_ple_reader_create.argtypes=[c.c_char_p,I,I,I,c.c_uint64,c.POINTER(c.c_int)];lib.gb10_ple_reader_create.restype=V
 for f in ['gb10_ple_reader_submit','gb10_ple_reader_submit_sqpoll','gb10_ple_reader_gather']:getattr(lib,f).argtypes=[V,V,I,V,c.c_uint]
 for f in ['gb10_ple_reader_complete','gb10_ple_reader_enable_sqpoll','gb10_ple_reader_destroy']:getattr(lib,f).argtypes=[V]
 libs[name]=lib
legacy=c.CDLL(str(P.parents[1]/'results/ple-reader-investigation-20260916/gather-legacy.so'));legacy.gb10_ple_gather.argtypes=[V,I,I,V,I,V,c.c_int]
rng=random.Random(183);records=[];width=160;nr=419430
with tempfile.TemporaryDirectory(prefix='resident-bench-') as td:
 p=pathlib.Path(td)/'table';data=rng.randbytes(nr*width);p.write_bytes(data)
 fd=os.open(p,os.O_RDONLY);os.fsync(fd)
 mapping=mmap.mmap(fd,0,access=mmap.ACCESS_COPY);view=(c.c_char*len(data)).from_buffer(mapping)
 for count in [64,768,1536]:
  for cache in [0,16*1024*1024]:
   handles={}
   for name,lib in libs.items():
    error=c.c_int();h=lib.gb10_ple_reader_create(os.fsencode(p),nr,width,131072,cache,c.byref(error));assert h,error.value;assert lib.gb10_ple_reader_enable_sqpoll(h)==0;handles[name]=h
   for condition in ['warm','cold']:
    samples={k:[] for k in ['baseline','resident','legacy1','legacy16']}
    for rep in range(32):
     ids_list=rng.sample(range(nr),count);ids=(I*count)(*ids_list);out=c.create_string_buffer(count*width)
     expected=b''.join(data[i*width:(i+1)*width] for i in ids_list)
     order=list(samples);rng.shuffle(order)
     for name in order:
      if condition=='cold':
       mapping.madvise(mmap.MADV_DONTNEED);os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
      else:
       for i in ids_list:_=mapping[i*width:(i+1)*width]
      start=time.perf_counter_ns()
      if name.startswith('legacy'):code=legacy.gb10_ple_gather(view,nr,width,ids,count,out,int(name[6:]))
      else:
       lib=libs[name];h=handles[name];code=lib.gb10_ple_reader_submit_sqpoll(h,ids,count,out,10000);assert code==0;code=lib.gb10_ple_reader_complete(h)
      duration=(time.perf_counter_ns()-start)/1e6
      assert code==0 and out.raw==expected,(name,code)
      if rep>=4:samples[name].append(duration)
    for name,vals in samples.items():
     record=dict(rows=count,cache=cache,condition=condition,variant=name,median_ms=statistics.median(vals),p95_ms=sorted(vals)[int(len(vals)*.95)],n=len(vals));records.append(record);print(json.dumps(record),flush=True)
   for name,h in handles.items():assert libs[name].gb10_ple_reader_destroy(h)==0
 del view;mapping.close();os.close(fd)
(R/'native-benchmark.json').write_text(json.dumps(records,indent=2)+'\n')
