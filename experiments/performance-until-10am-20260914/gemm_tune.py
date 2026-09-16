"""Run only in a coordinated idle window. Fixed K64, no per-request GEMV."""
import json,time,urllib.request,signal
from pathlib import Path
import torch
from vllm.model_executor.determinism.batch_invariant import matmul_kernel_persistent
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units
from triton.testing import do_bench
# Kill this process itself on deadline; timing out a docker client alone can leave an exec running.
signal.alarm(210)
out=Path('/tmp/gb10-gemm-tuning.json');sms=num_compute_units(0);torch.manual_seed(1701)
configs=[(128,128,8,3),(16,64,4,3),(16,128,4,3),(32,128,4,3),(64,64,4,3),(64,128,4,3)]
shapes=[('hc_down',336,10240),('hc_up',10240,320),('gdn_qkvz',8192,2560),('gdn_ba',48,2560),('gdn_out',2560,3072)]
rows=[]
def idle():
 with urllib.request.urlopen('http://127.0.0.1:30001/metrics',timeout=3) as r:raw=r.read().decode()
 assert all(float(l.rsplit(' ',1)[1])==0 for l in raw.splitlines() if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))),'Serving workload active: timing refused'
def mul(a,b,config):
 bm,bn,warps,stages=config;m,k=a.shape;n=b.shape[1];c=torch.empty((m,n),device=a.device,dtype=a.dtype)
 matmul_kernel_persistent[(min(sms,triton.cdiv(m,bm)*triton.cdiv(n,bn)),)](a,b,c,None,m,n,k,a.stride(0),a.stride(1),b.stride(0),b.stride(1),c.stride(0),c.stride(1),NUM_SMS=sms,A_LARGE=False,B_LARGE=False,C_LARGE=False,HAS_BIAS=False,BLOCK_SIZE_M=bm,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=64,GROUP_SIZE_M=8,num_stages=stages,num_warps=warps)
 return c
idle()
for name,n,k in shapes:
 weight=torch.randn((n,k),device='cuda',dtype=torch.bfloat16);b=weight.t();reference_x=torch.randn((1,k),device='cuda',dtype=torch.bfloat16)
 for config in configs:
  idle();reference=mul(reference_x,b,config)
  for count in [1,4,32,1600]:
   a=torch.randn((count,k),device='cuda',dtype=torch.bfloat16);positions=sorted({0,count//2,count-1});a[positions]=reference_x
   output=mul(a,b,config);torch.cuda.synchronize();exact=all(torch.equal(output[i],reference[0]) for i in positions)
   idle();ms=do_bench(lambda:mul(a,b,config),warmup=10,rep=30)
   rows.append({'shape':name,'N':n,'K':k,'M':count,'config':config,'fixed_k':64,'batch_exact':exact,'milliseconds':ms});out.write_text(json.dumps(rows,indent=2));print(rows[-1],flush=True)
   assert exact,'Batch-dependent arithmetic in candidate'
print('Finished; timings are random-tensor kernel measurements, not full-model throughput.',flush=True)
