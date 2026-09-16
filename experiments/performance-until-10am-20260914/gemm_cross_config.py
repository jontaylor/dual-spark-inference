"""Correctness only: no latency claim while real workload runs."""
import signal,json,time
from pathlib import Path
signal.alarm(60)
source=Path('/tmp/gemm_tune.py').read_text();exec(source.split('idle()\nfor name,n,k in shapes:')[0]);signal.alarm(60)
rows=[];start=time.time()
for name,n,k in [('hc_down',336,10240),('gdn_ba',48,2560)]:
 torch.manual_seed(481)
 weight=torch.randn((n,k),device='cuda',dtype=torch.bfloat16);b=weight.t();x=torch.randn((1,k),device='cuda',dtype=torch.bfloat16)
 ref=mul(x,b,(128,128,8,3))
 for count in [1,4,32,64,128,1600]:
  a=torch.randn((count,k),device='cuda',dtype=torch.bfloat16);positions=sorted({0,count//2,count-1});a[positions]=x
  baseline=mul(a,b,(128,128,8,3));candidate=mul(a,b,(64,64,4,3));equal=torch.equal(baseline,candidate)
  prefix=all(torch.equal(candidate[i],ref[0]) for i in positions)
  rows.append({'shape':name,'M':count,'all_outputs_equal_between_tiles':equal,'row_equals_baseline_C1':prefix});assert equal and prefix
result={'start':start,'end':time.time(),'scope':'Correctness only, overlapping serving workload. No timing claim.','rows':rows};Path('/tmp/gb10-cross-config.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
