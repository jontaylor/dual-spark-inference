"""Per-request draft scaling with a graph-stable request-state pointer."""
import torch
from vllm.triton_utils import triton,tl

@triton.jit
def _scale_draft(logits,out,idx,scales,STRIDE:tl.constexpr,V:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);offset=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    slot=tl.load(idx+row).to(tl.int64)
    beta=tl.load(scales+slot,mask=slot>=0,other=1.0).to(tl.float32)
    value=tl.load(logits+row*STRIDE+offset,mask=offset<V,other=0).to(tl.float32)
    if beta!=1.0:value=tl.div_rn(value,beta)
    tl.store(out+row*V+offset,value,mask=offset<V)

def scale_draft_logits(logits,idx_mapping,multipliers):
    # Output dtype is identical to logits. Gumbel sampling and verifier cache
    # therefore consume exactly the same representable pre-temperature values.
    out=torch.empty_like(logits,memory_format=torch.contiguous_format)
    n,v=logits.shape
    _scale_draft[(n,triton.cdiv(v,4096))](logits,out,idx_mapping,multipliers,STRIDE=logits.stride(0),V=v,BLOCK=4096)
    return out
