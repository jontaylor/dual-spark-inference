# Did our patches cause concurrency-dependent greedy output?

## Established findings

Our patches did not introduce the first measured numerical discrepancy. The pristine official vLLM image reproduces both the synthetic BF16 GEMM discrepancy and the real-checkpoint HC mixer discrepancy. The latter matches the recorded patched-server tensors bit-for-bit for serial and mixed batches.

Disabling our optional BF16 and MoE kernels, PLE acceleration, custom RoCE all-reduce and FLA dispatch override does not eliminate whole-response divergence: six serial responses match and all four concurrent responses differ. Those optimizations are not necessary for this effect.

The official-image serving tests also produce divergent responses with the fork's execution patches absent. They show additional serial instability: each ordinary/fixed-seed run produced four distinct serial outputs and four distinct concurrent outputs. The no-prefix-reuse, fixed-seed run produced three serial outputs and two concurrent outputs; none of its concurrent outputs matched any serial output. These are not clean reproductions of the exact serial-stable signature, but they establish response nondeterminism without the fork's execution patches. Combined with the exact pristine mixer replay, the confirmed underlying numerical issue is upstream, not introduced by our performance patches.

This does not prove every patch is free of numerical effects or quantify amplification. Differences in text across configurations do not measure model quality or regression severity. We did not localize the additional upstream serial instability to a particular kernel; the native QSA backport differs between the builds, but attributing the instability to it would require another controlled test.

## Controls and provenance

Same two GB10 GPUs (SM121), TP2/EP, NVIDIA Qwen3.8-Flash-Next-NVFP4 revision fc694b54fb0174e0913e6adf86691ef85a4ead47, archived 1930-token request, temperature zero, speculation disabled. Source version vLLM 0.29.0; PyTorch 2.13.0+cu130, CUDA 13.0. Serial controls bracket concurrency-four requests within each process lifetime. Prompt IDs are checked and phase boundaries require an idle API scheduler.

The pristine image is the official base pinned in the GB10 Dockerfile: `vllm/vllm-openai@sha256:18372a7224938643461b846fb64c5c9d3d6e9727e82caf2dc3043e620c9d4d7a`; resolved ARM64 image `sha256:785fd756b4ae1cacda7612cb859dbf227330e8ced9a6fdea42e9379128bebce0`. It was already available locally on both nodes. No new image was built for these tests.

`hyperconnection.py`, `ops/hc.py`, and the standard `layers/linear.py` match upstream v0.29.0 commit 98dff2a81d747d1dba01a47f939f48c3526d4206 byte-for-byte. Our optional low-latency dispatch adds measured GB10 plans, but the relevant HC shape [N=336,K=10240] only has plans for M=1,2,4,8. The measured M=330 and M=991 calls both fall back to normal linear. See `source-provenance.json` and the fork source.

## Standalone pristine tests

Neither standalone test uses our source overrides, model service, scheduler, paging, distributed reductions, or PLE implementation.

- Synthetic BF16 X=[330,10240], W=[336,10240], packed X=[991,10240]: 207 changed outputs, max 0.03125. Serial repeat exact. This is identical to the patched-image result. Fixed-schedule Triton and FP32-then-BF16 both eliminate differences in this small test.
- Actual HC checkpoint replay: serial and mixed outputs both exactly match captured serving outputs. The first projection changes 33 elements over 30 token rows, max 0.5; block-input output changes 51 elements over 21 token rows, max 0.0078125. The replay computes from checkpoint weights and captured token IDs; recorded hidden outputs are comparison targets only.

Evidence: `pristine-repro.json`, `pristine-hc-replay.json`, their logs, and the original scripts under ../gdn-trace-20260913/.

## Serving comparisons

| Configuration | Requests | Serial distinct | Concurrent distinct | Concurrent matching serial | Errors |
|---|---:|---:|---:|---:|---:|
| Existing patched configuration | 10 | 1 | 4 | 0 | 0 |
| Optional compute/transport optimizations off | 10 | 1 | 4 | 0 | 0 |
| Official image + checkpoint loader adapter | 10 | 4 | 4 | 0 (any serial) | 0 |
| Same official image, fixed seed | 10 | 4 | 4 | 0 (any serial) | 0 |
| Same official image, fixed seed, unique cache salts | 10 | 3 | 2 | 0 (any serial) | 0 |

The compute-off test retains paging, scheduling and PLE compatibility code; it is not represented as pristine upstream. `config.before.json`, `config.compute-off.json`, `baseline/`, `compute-off/` and `summary.json` retain settings and raw evidence.

## Official-image serving comparison design

The official serving run removes all fork scheduler, pager, CPU PLE/offload, transport, compute, model-runner and native QSA patches. PLE runs through upstream GPU embedding and communication code. No GB10 environment variables are set. The same inference/model arguments are used, except removal of the custom scheduler and KV connector. GPU cache budget, model length, batch limit, block size, synchronous scheduling, TP2, graph settings and original request are preserved.

One narrowly scoped checkpoint loader adapter is required: upstream only recognizes FP8 PLE for Fp8Config, whereas this checkpoint declares its FP8 PLE table through mixed ModelOpt configuration. The adapter reads that declared format and selects upstream's existing Qwen4ExpPLEFp8EmbeddingMethod. No arithmetic kernel or state management is changed. This is not described as a zero-patch full-model run. `upstream-ple-compat.patch` contains the complete adapter; preflight logs record the original selection (None) and corrected selection (FP8 method).

The first preflight tried importing both copies of the PLE module in one process, which failed duplicate custom-op registration. The subsequent preflight bind-mounted the adapter as the single PLE module and passed; this was a test-harness issue and did not require changing the adapter's computation.

`launch-upstream.py`, `upstream-command-r*.json` and `upstream-runtime-provenance.json` record the experiment. Completed full-model findings and service state are recorded below.

## Completed comparison and conclusion

All 50 serving requests succeeded, with identical prompt token IDs within each run. The no-cache run used the original model input and sampling settings plus a unique `cache_salt` per request; all ten returned zero cached prompt tokens, and all salts are archived in their per-request records. It did not eliminate upstream serial instability. Raw SSE, selected token IDs, top log probabilities, usage and generated bodies are saved under `baseline/`, `compute-off/`, `upstream/`, `upstream-seed/`, and `cold-upstream-seed/`.

The official run successfully loaded the same model (62.73 GiB per rank), allocated the unchanged 40 GiB KV budget, warmed up, and served the probes. Both rank image IDs and mount inventories were verified. The only vLLM source mount is the complete, archived PLE loader adapter. No model code was reloaded mid-test, and no request executed tools.

Answer to the attribution question: **our patches did not introduce the confirmed first batch-dependent numerical discrepancy; it reproduces exactly in the untouched official image. Our optional performance patches are not necessary for the full response divergence. Removing all fork execution patches also fails to restore deterministic responses.** A rollback of those patches is therefore not an evidenced fix. This finding does not certify all patches as numerically invariant or establish the relative contribution of each to the final output differences.

Final configuration: both deployment files have been restored byte-for-byte from their per-node pre-test backups, including the original compute optimizations and FLA override. The official-image test containers are being stopped and the managed original service is being resumed in the background. Speculation remains disabled. Post-restoration API readiness is not claimed; recovery is not a fix for the known divergence.
