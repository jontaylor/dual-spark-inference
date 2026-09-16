"""Scoped BF16 GEMM with a fixed K reduction schedule on GB10."""
import torch
from vllm.model_executor.determinism.batch_invariant import matmul_kernel_persistent
from vllm.triton_utils import triton
from vllm.utils.platform_utils import num_compute_units

# No M-indexed tuning: correctness baseline shared by prefill and decode.
GEMM_CONFIG = dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                   GROUP_SIZE_M=8, num_stages=3, num_warps=8)

def fixed_bf16_matmul(a, b, bias=None):
    assert a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype == torch.bfloat16 and a.is_cuda and b.is_cuda
    assert bias is None or (bias.ndim == 1 and bias.shape[0] == b.shape[1])
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    sms = num_compute_units(a.device.index)
    # Shape-specific tuning, constant across every batch size. K remains64.
    # Only the traced HC-down and GDN-ba projections opt into this plan.
    config = (dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
                   GROUP_SIZE_M=8, num_stages=3, num_warps=4)
              if (n, k) in {(336, 10240), (48, 2560)} else GEMM_CONFIG)
    grid = (min(sms, triton.cdiv(m, config["BLOCK_SIZE_M"]) * triton.cdiv(n, config["BLOCK_SIZE_N"])),)
    matmul_kernel_persistent[grid](
        a, b, c, bias, m, n, k, a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), c.stride(0), c.stride(1), NUM_SMS=sms,
        A_LARGE=a.numel() > 2**31, B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31, HAS_BIAS=bias is not None,
        **config)
    return c


from vllm.model_executor.layers.linear import UnquantizedLinearMethod

class FixedBF16LinearMethod(UnquantizedLinearMethod):
    def apply(self, layer, x, bias=None):
        return fixed_bf16_matmul(x, layer.weight.t(), bias)


HEAD_GEMM_CONFIG = dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
                        GROUP_SIZE_M=8, num_stages=3, num_warps=4)

def fixed_head_matmul(a, b, bias=None):
    assert a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype == torch.bfloat16 and a.is_cuda and b.is_cuda
    assert bias is None or (bias.ndim == 1 and bias.shape[0] == b.shape[1])
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=a.dtype)
    sms = num_compute_units(a.device.index)
    grid = (min(sms, triton.cdiv(m, 16) * triton.cdiv(n, 128)),)
    matmul_kernel_persistent[grid](
        a, b, c, bias, m, n, k, a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), c.stride(0), c.stride(1), NUM_SMS=sms,
        A_LARGE=a.numel() > 2**31, B_LARGE=b.numel() > 2**31,
        C_LARGE=c.numel() > 2**31, HAS_BIAS=bias is not None,
        **HEAD_GEMM_CONFIG)
    return c


from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

class FixedBF16EmbeddingMethod(UnquantizedEmbeddingMethod):
    def apply(self, layer, x, bias=None):
        return fixed_head_matmul(x, layer.weight.t(), bias)
