"""Synthetic SM121 BF16 batch-variance reproducer; no model or request data."""
import json
from pathlib import Path
import torch

torch.manual_seed(917352)
x=torch.randn(330,10240,device='cuda',dtype=torch.bfloat16)
w=torch.randn(336,10240,device='cuda',dtype=torch.bfloat16)*0.02
packed=torch.cat([x[:1],x,x,x],dim=0)
assert torch.equal(packed[1:331],x)

def compare(name,op):
    a=op(x,w)
    b=op(packed,w)[1:331]
    again=op(x,w)
    d=a.float()-b.float()
    return {'method':name,'identical_input_rows':True,
            'serial_repeat_equal':torch.equal(a,again),
            'cross_batch_equal':torch.equal(a,b),
            'changed_elements':int((d!=0).sum()),
            'max_abs':float(d.abs().max())}

results=[compare('torch BF16 linear',torch.nn.functional.linear)]
from vllm.model_executor.determinism.batch_invariant import matmul_persistent
results.append(compare('fixed-schedule Triton',lambda x,w:matmul_persistent(x,w.t())))
torch.backends.cuda.matmul.fp32_precision='ieee'
results.append(compare('FP32 linear then BF16',lambda x,w:torch.nn.functional.linear(x.float(),w.float()).bfloat16()))
out={'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),
     'torch':torch.__version__,'cuda':torch.version.cuda,'results':results}
print(json.dumps(out,indent=2))
Path('/tmp/minimal-gemm-results.json').write_text(json.dumps(out,indent=2))
