"""Independent file, verified cache state; never evict live serving data."""
import ctypes as c,json,mmap,os,pathlib,random,statistics,tempfile,time
os.environ['OMP_WAIT_POLICY']='PASSIVE';P=pathlib.Path(__file__).resolve().parent;R=P.parents[1]/'results/ple-resident-20260916';V=c.c_void_p;I=c.c_int64
libc=c.CDLL(None,use_errno=True);libc.mincore.argtypes=[V,c.c_size_t,V];libc.madvise.argtypes=[V,c.c_size_t,c.c_int]
libs={}
for name,path in [('current_sqpoll',P.parent/'ple-latency-campaign-20260916/variants/17-fixed-policy/libple_batch_reader.so'),('batched_prefetch',P/'libple_batch_reader.so')]:
 lib=c.CDLL(str(path));lib.gb10_ple_reader_create.argtypes=[c.c_char_p,I,I,I,c.c_uint64,c.POINTER(c.c_int)];lib.gb10_ple_reader_create.restype=V
 lib.gb10_ple_reader_submit_sqpoll.argtypes=[V,V,I,V,c.c_uint]
 for f in ['gb10_ple_reader_complete','gb10_ple_reader_enable_sqpoll','gb10_ple_reader_destroy']:getattr(lib,f).argtypes=[V]
 libs[name]=lib
legacy=c.CDLL(str(P.parents[1]/'results/ple-reader-investigation-20260916/gather-legacy.so'));legacy.gb10_ple_gather.argtypes=[V,I,I,V,I,V,c.c_int]
rng=random.Random(534);width=160;nr=419430;results=[]
with tempfile.TemporaryDirectory(prefix='ple-verified-cold-') as td:
 p=pathlib.Path(td)/'table';data=rng.randbytes(nr*width);p.write_bytes(data);fd=os.open(p,os.O_RDONLY);os.fsync(fd)
 mapping=mmap.mmap(fd,0,access=mmap.ACCESS_COPY);view=(c.c_char*len(data)).from_buffer(mapping);bits=(c.c_ubyte*((len(data)+4095)//4096))()
 for count in [64,768,1536,2048]:
  error=c.c_int();handles={}
  for name,lib in libs.items():
   h=lib.gb10_ple_reader_create(os.fsencode(p),nr,width,131072,0,c.byref(error));assert h,error.value;assert lib.gb10_ple_reader_enable_sqpoll(h)==0;handles[name]=h
  ids_list=rng.sample(range(nr),count);ids=(I*count)(*ids_list);out=c.create_string_buffer(count*width)
  expected=b''.join(data[i*width:(i+1)*width] for i in ids_list)
  def call(name):
   if name.startswith('legacy'):return legacy.gb10_ple_gather(view,nr,width,ids,count,out,int(name[6:]))
   lib=libs[name];code=lib.gb10_ple_reader_submit_sqpoll(handles[name],ids,count,out,10000)
   return code if code else lib.gb10_ple_reader_complete(handles[name])
  names=['current_sqpoll','batched_prefetch','legacy1','legacy16']
  for name in names:assert call(name)==0
  samples={n:[] for n in names};residency=[]
  for rep in range(24):
   order=names[:];rng.shuffle(order)
   for name in order:
    for line in pathlib.Path('/proc/self/maps').read_text().splitlines():
     if line.endswith(str(p)):
      lo,hi=[int(v,16) for v in line.split()[0].split('-')];assert libc.madvise(lo,hi-lo,mmap.MADV_DONTNEED)==0
    os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
    assert libc.mincore(c.addressof(view),len(data),bits)==0
    pages={page for row in ids_list for page in range(row*width//4096,((row+1)*width-1)//4096+1)}
    hot=sum(bool(bits[page]&1) for page in pages);residency.append(hot)
    assert hot==0,('not cold',hot,len(pages))
    start=time.perf_counter_ns();code=call(name);elapsed=(time.perf_counter_ns()-start)/1e6
    assert code==0 and out.raw==expected,(name,code)
    samples[name].append(elapsed)
  for name,values in samples.items():
   row={'rows':count,'variant':name,'n':len(values),'median_ms':statistics.median(values),'p95_ms':sorted(values)[22],'all_target_pages_verified_nonresident':not any(residency)};results.append(row);print(json.dumps(row),flush=True)
  for name,h in handles.items():assert libs[name].gb10_ple_reader_destroy(h)==0
 del view;mapping.close();os.close(fd)
(R/'verified-cold.json').write_text(json.dumps(results,indent=2)+'\n')
