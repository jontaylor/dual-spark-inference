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
out=Path('/tmp/gb10-fine-gemm-tuning.json');sms=num_compute_units(0);torch.manual_seed(1701)
configs=[(128,128,8,3),(16,16,4,3),(16,32,4,3),(32,32,4,3),(64,32,4,3),(64,64,4,3)]
shapes=[('hc_down',336,10240),('gdn_ba',48,2560)]
rows=[]
def idle():
 # This variant is reserved for the already-planned server-off deployment window.
 # Confirm no serving API before using the otherwise-unoccupied GPU.
 try:
  with urllib.request.urlopen('http://127.0.0.1:30001/health',timeout=1):pass
 except (urllib.error.URLError,TimeoutError):return
 raise RuntimeError('Serving API is live; offline benchmark refused')
def mul(a,b,config):
 bm,bn,warps,stages=config;m,k=a.shape;n=b.shape[1];c=torch.empty((m,n),device=a.device,dtype=a.dtype)
 matmul_kernel_persistent[(min(sms,triton.cdiv(m,bm)*triton.cdiv(n,bn)),)](a,b,c,None,m,n,k,a.stride(0),a.stride(1),b.stride(0),b.stride(1),c.stride(0),c.stride(1),NUM_SMS=sms,A_LARGE=False,B_LARGE=False,C_LARGE=False,HAS_BIAS=False,BLOCK_SIZE_M=bm,BLOCK_SIZE_N=bn,BLOCK_SIZE_K=64,GROUP_SIZE_M=8,num_stages=stages,num_warps=warps)
 return c
idle()
for name,n,k in shapes:
 weight=torch.randn((n,k),device='cuda',dtype=torch.bfloat16);b=weight.t();reference_x=torch.randn((1,k),device='cuda',dtype=torch.bfloat16)
 baseline_reference=mul(reference_x,b,configs[0])
 for config in configs:
  idle()
  try:
   reference=mul(reference_x,b,config)
   cross_exact=torch.equal(reference,baseline_reference)
   if not cross_exact:
    rows.append({'shape':name,'config':config,'eligible':False,'reason':'C1 differs across tile configurations'})
    out.write_text(json.dumps(rows,indent=2));print(rows[-1],flush=True);continue
   for count in [1,4,8,32,48,64,128,1600]:
    # Include real random rows as well as a repeated C1 row at distinct positions.
    a=torch.randn((count,k),device='cuda',dtype=torch.bfloat16);positions=sorted({0,count//2,count-1});a[positions]=reference_x
    baseline=mul(a,b,configs[0]);output=mul(a,b,config);torch.cuda.synchronize()
    exact=all(torch.equal(output[i],reference[0]) for i in positions)
    all_rows_exact=torch.equal(output,baseline)
    idle();ms=do_bench(lambda:mul(a,b,config),warmup=10,rep=30)
    row={'shape':name,'N':n,'K':k,'M':count,'config':config,'fixed_k':64,'batch_exact':exact,'cross_config_C1_exact':cross_exact,'cross_config_all_rows_exact':all_rows_exact,'eligible':exact and all_rows_exact,'milliseconds':ms}
    rows.append(row);out.write_text(json.dumps(rows,indent=2));print(row,flush=True)
    if not row['eligible']:break
  except Exception as error:
   # Unsupported fine tiles must not prevent testing remaining conservative plans.
   rows.append({'shape':name,'config':config,'eligible':False,'error':repr(error)})
   out.write_text(json.dumps(rows,indent=2));print(rows[-1],flush=True)
   if isinstance(error,torch.cuda.OutOfMemoryError):raise
print('Finished; inspect EVERY eligible flag before selecting a plan. Random-tensor kernel timings are not full-model throughput.',flush=True)
