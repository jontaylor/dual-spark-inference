from pathlib import Path
p=Path('/home/jon/vllm-gb10-kv-paging/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py')
s=p.read_text().replace('from vllm.triton_utils import tl, triton','from vllm.triton_utils import tl, triton\nfrom vllm.third_party.flash_linear_attention.ops.op import exp')
s=s.replace('b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))','b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)').replace('b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))','b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)').replace('b_h *= tl.exp(b_g)','b_h *= exp(b_g)').replace('num_warps = 4','num_warps = 1')
s=s.replace('        # Update pointers for next timestep','        # Match a cache write/read between ordinary decode steps.\n        b_h = b_h.to(ht.dtype.element_ty).to(tl.float32)\n\n        # Update pointers for next timestep')
Path(__file__).with_name('fused_sigmoid.fixed.py').write_text(s)
