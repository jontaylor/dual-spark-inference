"""GPU hash differential test, including padding, EOS, overflow and negatives."""
import ctypes as c,itertools,json,random,statistics,time
import torch
from vllm.v1.ple_offload.gb10_mapped import map_shared_tensor

torch.set_num_threads(1)
v=c.c_void_p;i=c.c_int64
cpu=c.CDLL('/opt/gb10/libple_gather.so').gb10_ple_hash;cpu.argtypes=[v,i,v,i,v,i,i,i,i,v,v,v,v];cpu.restype=c.c_int
lib=c.CDLL('/build/libple_hash_gpu.so');gpu=lib.gb10_ple_hash_gpu;gpu.argtypes=cpu.argtypes+[v,v];gpu.restype=c.c_int
stream=torch.cuda.current_stream();error=torch.zeros(1,dtype=torch.int32).share_memory_();error_dev=map_shared_tensor(error,torch.device('cuda:0'));error_ref=c.c_int32.from_address(error.data_ptr());event=torch.cuda.Event()
checked=0
for seed in range(300):
 rng=random.Random(seed);torch.manual_seed(seed)
 ngram=rng.choice([2,3,4,8]);heads=rng.choice([1,8,16]);reqs=rng.choice([1,2,4,12,32]);lens=[rng.randint(0,32) for _ in range(reqs)]
 if sum(lens)<reqs:lens[0]+=reqs
 count=sum(lens)+rng.randrange(17);eos=248044
 tokens=torch.randint(-4,248320,(count,),dtype=torch.int32)
 tokens[0]=-1
 history=torch.randint(0,248320,(reqs,ngram-1),dtype=torch.int32)
 for x in (tokens,history):x[torch.rand(x.shape)<.2]=eos
 query=torch.tensor([0]+list(itertools.accumulate(lens)),dtype=torch.int32)
 multipliers=torch.randint(1,2**62,(ngram,),dtype=torch.int64)
 nh=(ngram-1)*heads;sizes=torch.randint(1000,20000000,(nh,),dtype=torch.int64);offsets=torch.cat([torch.tensor([0]),sizes.cumsum(0)[:-1]])
 host_args=[tokens.to(torch.int64).clamp_min_(0),query.to(torch.int64),history.to(torch.int64)]
 ref=torch.empty((count,nh),dtype=torch.int64)
 assert cpu(host_args[0].data_ptr(),count,host_args[1].data_ptr(),reqs,host_args[2].data_ptr(),ngram-1,ngram,heads,eos,multipliers.data_ptr(),sizes.data_ptr(),offsets.data_ptr(),ref.data_ptr())==0
 dev=[x.cuda() for x in (tokens,query,history,multipliers,sizes,offsets)]
 out=torch.empty_like(ref).share_memory_();out_dev=map_shared_tensor(out,torch.device('cuda:0'))
 error_ref.value=0
 assert gpu(dev[0].data_ptr(),count,dev[1].data_ptr(),reqs,dev[2].data_ptr(),ngram-1,ngram,heads,eos,dev[3].data_ptr(),dev[4].data_ptr(),dev[5].data_ptr(),out_dev.data_ptr(),error_dev.data_ptr(),stream.cuda_stream)==0
 event.record(stream);event.synchronize()
 assert error_ref.value==0
 assert torch.equal(out,ref),(seed,torch.nonzero(out!=ref)[:10])
 # Out-of-order boundaries must be rejected, without any gathered result.
 if seed==0:
  dev[1][0]=1;error_ref.value=0;out.fill_(-1)
  assert gpu(dev[0].data_ptr(),count,dev[1].data_ptr(),reqs,dev[2].data_ptr(),ngram-1,ngram,heads,eos,dev[3].data_ptr(),dev[4].data_ptr(),dev[5].data_ptr(),out_dev.data_ptr(),error_dev.data_ptr(),stream.cuda_stream)==0
  event.record(stream);event.synchronize();assert error_ref.value==-1;assert torch.all(out==-1)
 checked+=1
print(json.dumps({'differential_cases':checked,'passed':True}),flush=True)
