# Scoped batch-invariant inference on GB10

**Status: fixed candidate deployed and healthy on both ranks; final normal-graph validation passed. Left running.**

The final deployment passed **56 serving requests** with exact token-ID and returned-score equality against the corresponding serial reference. Metrics confirmed **eight simultaneous running requests** in the mixed-prompt test. HC also passes C1/C4 bit-for-bit using one M=4 GEMM per projection.

## Final serving validation

| Test | Requests | Result |
|---|---:|---|
| Original prompt, serial3/concurrent4/serial3 | 10 | All 125-token responses and returned scores identical |
| Same, fixed seed | 10 | Identical to the first experiment |
| Unique cache salt per request | 10 | Identical; every request reports zero cached prompt tokens |
| Unique cache salts, fixed seed | 10 | Identical; every request reports zero cached prompt tokens |
| Four prompt lengths, serial references / eight staggered concurrent / serial afterward | 16 | All 96-token responses and returned scores match their prompt-specific reference |

Mixed prompt lengths were 36, 146, 1,826 and 3,526 tokens. Server metrics recorded peak running concurrency 8. Normal CUDA graphs remained enabled. All 40 original-prompt responses match across the four experiments; the mixed test compares each prompt separately. The mixed responses were bounded at 96 tokens, so this is not a claim about complete long generations.

Normal original-prompt C4 measured 500 generated tokens in 9.07 seconds (55.1 aggregate tokens/second including prefill), versus median serial response time 7.52 seconds. This demonstrates concurrent operation; it is not a controlled before/after throughput benchmark.

Evidence: `final-original-validation.json`, `final-mixed-validation.json`, `probe-summary.json`, `normal-final/`, `normal-final-seed/`, `uncached-final/`, `uncached-final-seed/`, and `mixed-20260914T003356Z/` (including raw score records and concurrency metrics). `final-deployment-check.json` verifies both live configs and all 13 runtime source hashes on each rank. Final API health returned HTTP 200.

Earlier prospective eager tests passed 20 complete 117-token responses. Normal-graph responses contain 125 tokens; cross-execution-mode or cross-restart equality is not claimed.

## Changes

All GEMMs remain batched. No global `VLLM_BATCH_INVARIANT` switch and no per-request serialization.

| Operation | Scoped change |
|---|---|
| HyperConnection projections | Persistent Triton GEMM, fixed128/128/64 M/N/K tiles |
| GDN input/output BF16 projections | Same persistent GEMM, preserving linear-layer loading and TP behavior |
| GDN gated RMS norm | Fixed ROWS_PER_BLOCK=1; consistent explicit norm path |
| GDN mixed decode | Same packed recurrent arithmetic as pure decode |
| GDN prefill | use_cp=False for FlashInfer; canonical 64-token nonfinal prefill grid |
| QSA BF16 QKV/output/index projections | Same fixed GEMM |
| QSA sparse attention | Fixed BLOCK_N64, target splits8, warps2; no row-count-dependent reduction schedule |
| MoE router and shared-expert BF16 projections, including scalar gate | Same fixed GEMM; quantized routed-expert kernels retained |
| PLE key/value BF16 projections | Same fixed GEMM; PLE offload/table/transport retained |
| Final LM head | Fixed16/128/64 tiles, 4 warps, 3 stages; embedding-method inheritance retained |

BLOCK_SIZE_K remains 64 for every patched BF16 projection, regardless of M. The smaller head tile matched the128-row fixed kernel bit-for-bit and passed C1/C4/C8. Its isolated GB10 time was2.82/2.84/2.87ms, compared with stock cuBLAS3.72/2.70/2.81ms. These are kernel timings, not whole-model speedups.

Speculation remains disabled. TP2/EP, request capacity 32, normal CUDA graphs, the 1600-token cache block and existing RoCE/PLE configuration are preserved.

## Evidence chain

1. HC C1 versus one M4 call per projection passes bit-for-bit. Recorded batched GEMMs are M4/K10240/N336 and M4/K320/N10240. Captured M330/M991 HC replay also matches.
2. Fixed GDN projections and gated norm remove observed primitive differences. Full serving traces then match GDN0/2 across every captured prefill row.
3. Next first difference was QSA layer3 output with equal input. Fixed projections and sparse-attention reduction remove it. QSA index scoring/top-k checks passed unchanged.
4. All-row/eager tracing exposed MoE router/shared-expert/PLE projection differences and final logits with equal hidden states. Fixed GEMMs remove their primitive differences.
5. A last-token trace appeared to implicate GDN45. Full-row hashes instead located an earlier scalar gate difference in layer44. Actual captured M330/M991 inputs match; stock scalar GEMM changes row122 by0.015625, fixed GEMM changes zero values. This explains propagation through GDN45.
6. Prospective scalar-gate fix: all20 complete responses match117 token IDs and all reported scores, across both seed experiments.
7. Permanent normal-graph deployment passes all 56 final serving requests described above, including uncached repeats and mixed C8.

## Execution and cache checks

Real convolution plus recurrent routing passes with actual per-rank H8/HV24, for FP32 and BF16 recurrent states. Complete GDN core checks (real convolution, gating and recurrent prefill) pass C1/C4 and one-decode-plus-three-prefill batches at330 tokens per prefill, with synthetic inputs and valid cache slots.

A192-token Triton prefill matches64+64+64 in output and state; negative control73+55+64 differs. The scheduler keeps nonfinal stops on64-token boundaries while retaining1600-token cache materialization. Completion lookup skips snapshots whose boundary is not divisible by64; this reduces completion-cache hit coverage. Aligned lookup and pending-record behavior have logic regression checks. This is not a claim of cold-versus-warm equivalence for every multi-turn completion-cache or paging scenario.

PLE normalization/gate-sum checks passed60 captured-row replications; these operations remain unchanged.

## Source and attribution

Adapted execution-shape ideas from [vLLM #49827](https://github.com/vllm-project/vllm/pull/49827), head debb4ba43485a3efa41e6c81180e004b81be6a16. Inspected [#45819](https://github.com/vllm-project/vllm/pull/45819), head c6cb0851075013a0e14cae08bad688c2bb83f055; its per-sequence projection/recurrence loops were not ported. Both PRs were open when inspected. Their reported validation does not cover this entire TP2/EP/quantized/cached deployment.

The original QSA reduction source matches upstream model-support commit e126687a9a (#53896), SHA256 faa8d358c79745f304edd363e4da21992e4cf015a22316b14980500bd199a0ad. Our local top-k backport did not introduce its M-dependent split schedule. Earlier official-image HC/cuBLAS attribution is recorded in ../patch-attribution-20260913/REPORT.md.

## Reproduction and deployment

Final candidate configs: candidate-final-r0.json and candidate-final-r1.json. Original per-rank configs: config-before-r0.json and config-before-r1.json. Runtime overrides are SHA256-verified. Source changes and separate arithmetic/execution patches are in this directory; STATUS.md records the investigation and intermediate candidates.

Key primitive evidence: projection-replay.json, gdn-execution-patched.json, gdn-execution-baseline.json, scheduler-checks.json, completion-checks.json, qsa-attention-check.json, qsa-index-check.json, dense-projection-check.json, head-projection-check.json, head-benchmark.json, ple-norm-check.json, scalar-projection-check.json.

Prospective serving evidence: scalar-pilot/, scalar-128/, scalar-128-seed/. Raw SSE, prompt/generated IDs, score records and phase metrics are preserved. No tools are executed by these probes.

Diagnostics are opt-in through local container control files. They can capture all-row hashes, selected full tensors and eager decode. Normal requests retain graph dispatch. Diagnostic-only extra BF16 methods are restored when control is disabled; the final scalar fix is in model initialization and does not depend on that mechanism.

## Remaining scope and tradeoffs

Speculation is disabled. Completion-cache snapshots outside the 64-token grid are skipped, reducing cache-hit coverage. This validation does not establish exhaustive cold/warm multi-turn cache equivalence, disk-pressure paging invariance, stochastic-sampling invariance, or behavior at every concurrency and prompt length. Request capacity remains 32, but the measured end-to-end concurrency peak here is 8. No global GEMM replacement or per-request serialization was introduced.
