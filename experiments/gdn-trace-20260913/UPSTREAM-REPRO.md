# SM121 BF16 linear changes with batch shape despite identical rows

Environment: NVIDIA GB10 (compute capability 12.1), PyTorch 2.13.0+cu130, CUDA 13.0. vLLM 0.29.0 provides the optional comparison kernel; it is not required to reproduce the baseline.

Run `python3 minimal_gemm_repro.py` in the serving image with one visible GPU. It writes `/tmp/minimal-gemm-results.json`. The script uses seeded synthetic random matrices and contains no model weights or request data.

For BF16 `F.linear`, compare `[330,10240] @ [10240,336]` with the same input rows embedded at rows 1:331 of a `[991,10240]` matrix. Observed: 207 changed elements, max absolute difference 0.03125; repeated serial invocation is bit-identical. Fixed-schedule Triton and FP32 linear followed by BF16 casting each give zero changed elements for the tested shapes.

An additional real-model projection trace selected the same named cuBLAS split-K kernel at both row counts, but launch grid changed from `[24,1,4]` to `[24,1,2]`, followed by `cublasLt::splitKreduce_kernel`. Kernel name: `nvjet_sm121_tst_mma_128x128x64_3_64x32x64_tmaAB_splitK_TNNN`.

Stock vLLM batch-invariant initialization did not eliminate this real-projection discrepancy on SM121. Its persistent aten matmul overrides are gated to device capability family 80; adding family 120 eliminated the discrepancy in an isolated mixer test. Full-server use is separately blocked for the model by vLLM's explicit unsupported `GDN_ATTN` guard. This report requests investigation/coverage, not a claim that a one-line architecture extension supports every model.

Suggested upstream split: report the synthetic BF16 shape dependence and split-K geometry to PyTorch/cuBLAS maintainers; report SM12x batch-invariance coverage plus the separate GDN backend limitation to vLLM. No issue was posted automatically.
