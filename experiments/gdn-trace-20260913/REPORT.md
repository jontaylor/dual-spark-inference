# Concurrency-dependent greedy inference: diagnosis and remediation

## Finding

A confirmed source of batch-dependent computation is the first HyperConnection mixer’s BF16 merged down/injection projection. This is upstream of recurrent attention and sampling. Replaying checkpoint weights in a separate single-GPU process reproduces the captured serial and mixed-batch mixer outputs bit-for-bit, without serving state, request routing, KV cache, or inter-node communication.

This identifies a concrete numerical source; it does not establish that it is the only batch-sensitive operation in the model. Large eventual logit differences alone did not establish a state/indexing bug.

## Environment and controls

Two NVIDIA GB10 nodes (SM121), TP2 + EP; vLLM 0.29.0, PyTorch 2.13.0+cu130, CUDA 13.0. NVIDIA Qwen3.8-Flash-Next-NVFP4 revision fc694b54fb0174e0913e6adf86691ef85a4ead47. Custom paging, BF16 KV, CPU-offloaded PLE, GB10 kernels, and RoCE reductions.

Speculation was disabled (`mtp_tokens=0`, per-request speculation metrics disabled). Explicit block size 1600 preserves the prior effective block size and retention alignment. Identical archived temperature-zero requests still diverged at concurrency four, including fixed-seed tests. Serial controls were repeated within each server lifetime because serial output sometimes changed across restarts.

## Evidence chain

1. Request indices, input token IDs, positions and effective GPU temperatures agreed with requests. Raw logits and selected tokens mapped correctly to returned API tokens. Earlier traces passed 612 slot checks with no writable-block aliases. Both ranks matched final hidden/logit tensors across 72 captured steps. See ../row-trace-20260913/REPORT.md and associated traces.
2. A representative mixed forward scheduled `[1, 330, 330, 330]` tokens after computed-token counts `[1930, 1600, 1600, 1600]`, for 991 total rows. The serial comparison processes the same 330 prompt-tail tokens alone.
3. Full captures show identical initial convolution/recurrent states in GDN layers 0 and 2. Layer-0 input already differs at 21 prompt positions. This supersedes the earlier last-token-only attribution to GDN output.
4. Checkpoint mixer replay: grouped Gemma RMSNorm outputs are identical. The merged BF16 projection is the first measured divergence: 33 changed elements across 30 token rows, max absolute difference 0.5. Final mixer output differs in 51 elements across 21 token rows, max 0.0078125. Baseline replay exactly matches both serving tensors.
5. The profiler records `nvjet_sm121_tst_mma_128x128x64_3_64x32x64_tmaAB_splitK_TNNN`, followed by cuBLASLt split-K reduction. GEMM grids change from `[24,1,4]` at M=330 to `[24,1,2]` at M=991. Batch-dependent split-K execution/accumulation precedes BF16 rounding.
6. A standalone synthetic reproduction with BF16 random matrices X=[330,10240], W=[336,10240], and packed X=[991,10240] changes 207 elements (max 0.03125), while serial repeats are exact. It needs no checkpoint or private request data.

## Tested remediation options

| Option | Observed result | Scope / tradeoff |
|---|---|---|
| One shared inference gate serializing callers | 10/10 full responses identical, including four queued callers | Verified immediate workaround for this workload; all inference traffic must share the gate. Reduces concurrency and throughput. Not installed as a production gate. |
| Disable speculation / fix seed | Divergence persists | Useful isolation controls; do not solve this issue. |
| Stock `VLLM_BATCH_INVARIANT=1` | Real-checkpoint mixer still differs | This vLLM version installs persistent matmul overrides only for SM80. Its cuBLAS workspace fallback is insufficient here on SM121. |
| Disable BF16 reduced-precision reduction | Mixer still differs | Not an effective remedy in this test. |
| FP32 mixer projections, cast results to BF16 | Mixer and synthetic reproduction become identical across tested shapes | Local numerical workaround; full-model determinism and performance are unverified. |
| Fixed-schedule Triton projections | Mixer and synthetic reproduction become identical across tested shapes | Strong kernel-level candidate; requires coverage of all relevant model operations and performance validation. |
| Extend vLLM persistent-matmul overrides to SM12x | Passes isolated mixer through normal batch-invariance initialization | Full server fails startup: `VLLM batch_invariant mode is not supported for GDN_ATTN`. Prototype, not a deployable fix for this model. |

The ungated full-service sequence produced serial hash `a9b3fa8a9358` and four different concurrent hashes (`d1934bdb1298`, `02c5728c96be`, `047357b9cff6`, `74774475ea46`). The gated sequence produced the serial hash for all ten successful responses. Hashes exclude random tool-call IDs and compare generated reasoning, content and tool names/arguments.

## Artifacts

- `hc_replay.py`, `hc-replay-*.json`: checkpoint replay and alternative arithmetic paths.
- `hc-kernels.json`, `hc-kernel-names.json`: profiler and launch geometry.
- `minimal_gemm_repro.py`, `minimal-gemm-results.json`: synthetic reproducer suitable for an upstream issue (no request data).
- `batch_invariant.sm12x.py`, `sm12x-batch-invariant.patch`: prototype architecture extension.
- `traces-r0/`, `traces-r1/`, `weights-r0/`, `weights-r1/`: full-input/state evidence.
- `uncaptured/`, `captured/`, `serialized/`: original API responses, token IDs and request payloads. These include user prompt data; use the synthetic reproducer for public sharing.
- `../batch-invariant-20260913/`: full-server prototype configuration, backups and validation evidence.

## Recommendation

Use a shared serialization gate where repeatability is required immediately. For concurrent deterministic serving, pursue SM12x fixed-schedule GEMM support upstream and validate the entire quantized/recurrent model, including custom kernels. The minimal reproducer and cuBLAS launch evidence give a focused upstream report. FP32 projections are another targeted engineering option; neither primitive-level success nor this single workload proves general determinism.

## Full-server prototype result

The architecture-extension prototype and `VLLM_BATCH_INVARIANT=1` were installed on both nodes, with trace hooks removed and all other serving settings preserved. Model weights loaded, then initialization failed in `vllm/v1/attention/selector.py::_cached_get_mamba_attn_backend` with `RuntimeError: VLLM batch_invariant mode is not supported for GDN_ATTN`. No full-service probe could run against this candidate. See `../batch-invariant-20260913/startup.log` and `startup-r1.log`.

The kernel-level SM12x extension is insufficient to enable global deterministic serving of this model. A focused mixer-only implementation could avoid the global mode restriction, but would still require validation of recurrent attention, quantized MoE and other kernels. The guard was not bypassed to claim unsupported determinism.

User requested leaving a successful fix running. This candidate did not qualify: the clean configuration and original launcher were restored on both nodes. Speculation remains disabled; no production serialization gate was installed. The baseline reload was subsequently superseded by the targeted HC candidate below; no claim of a successful concurrent-serving fix is made.

## Targeted HC fix without global mode

A second candidate replaces only HyperConnection down/injection and up projections with `matmul_persistent`, using the original vLLM modules for checkpoint weight loading. It does not enable global batch-invariant mode or change attention backends. The actual patched `GatedResidual.mix` class was replayed independently; both block-input and injection tensors are bit-identical across the three corresponding mixed-batch slices.

The full server successfully starts and serves this targeted candidate. Original payload test: 10 successful responses; all six serial controls match, but all four concurrent responses differ from serial and each other. All four first differ at generated token 3. Thus the confirmed first mixer source is not the only source requiring remediation. Removing it is insufficient for full-model determinism. This candidate is not classified as a successful fix.

Evidence: `../hc-fixed-gemm-20260913/` contains the patch, actual-class replay, configurations, raw request/response captures, and `summary.json`. The current fixed-seed result and final runtime state are appended when recorded.

Final state: the unsuccessful targeted candidate was removed from both deployment configurations. Recovery of the prior serving configuration is running in the background; API readiness has not been re-verified after this last restart. Speculation remains off. No production serialization gate or successful concurrent-serving fix is installed.
