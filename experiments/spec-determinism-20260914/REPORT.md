# Deterministic concurrent inference with three-token MTP

**Deployed, validated, and left running.** The existing background temperature trials have been resumed with their original process state and sampling settings.

## Results

At temperature zero, **86 requests passed exact token-ID and returned-score comparisons against their respective serial references**. Metrics confirmed **eight simultaneous running requests**. Across 9,596 returned tokens, every selected token was a reported score maximum.

| Probe | Requests | Result |
|---|---:|---|
| Short serial3 / concurrent4 / serial3 | 10 | All four-token outputs and scores match |
| 128-token limit, ordinary and fixed seed | 20 | All tokens and scores match |
| Complete responses, ordinary and fixed seed | 20 | All 145-token outputs and scores match |
| Unique cache salt per request, ordinary and fixed seed | 20 | All 128-token outputs and scores match; zero cached prompt tokens |
| Four prompt lengths, serial references / C8 / serial afterward | 16 | All 96-token outputs and scores match their prompt-specific reference |

Token and score prefixes also match across seed settings, cache salts and generation limits. All 70 original-prompt requests reported zero cached prompt tokens. Mixed prompts contain 36, 146, 1,826 and 3,526 tokens. The C8 probe uses staggered arrivals to overlap decode and prefill.

## Speed

| Measurement | Previous deterministic no-spec deployment | Current MTP3 deployment |
|---|---:|---:|
| Median complete-response latency, same original prompt | 7.52 seconds, 125 generated tokens | 3.89 seconds, 145 generated tokens |
| C4 aggregate generation throughput including prefill | 55.1 tokens/second | 90.8 tokens/second |

The fixed-seed repeat measured 3.88 seconds and 90.9 tokens/second. The 128-token probes measured about 3.51 seconds serially. These are indicative comparisons under isolated probe load, not identical-output before/after benchmarks or promises for every workload. Normal CUDA graphs and request capacity 32 remain enabled; no request serialization or per-request GEMV was introduced.

Live counters confirm speculation is active: 2790 draft steps, 8370 proposed tokens, 6834 accepted tokens (81.6% acceptance) during the probes. See speculation-summary.json.

## Changes

The prior scoped fixed-K persistent GEMMs, fixed GDN gated RMS norm, canonical 64-token prefill grid, QSA reduction and projection fixes remain in place.

A new GDN speculative recurrent kernel uses the same normalization, exponential and reduction arithmetic as the existing packed ordinary decode kernel, with a fixed one-warp configuration. It rounds each intermediate state to the cache dtype before advancing to the next token. This matches the cache write/read rounding in ordinary decoding. The temporal loop runs inside the GPU kernel; requests and heads remain parallel.

The unmodified speculative kernel passes batch replication, but with BF16 state its four-token output differs from four single-token calls. The candidate removes that discrepancy and also matches the packed decode path.

The scheduler startup guard now permits the tested MTP configuration with up to three draft tokens. Other hidden-state speculative methods remain rejected. The GDN module imports the new recurrence explicitly; unrelated model kernels retain their implementations.

## Kernel and state validation

Using actual GB10 per-rank head dimensions and synthetic inputs, the candidate passes bit-for-bit outputs and states for:

- C1 versus C4 recurrence, with FP32 and BF16 cache state.
- Four-token verification versus four single-token calls and versus packed ordinary decode.
- Real convolution plus recurrence, alone and C4.
- Rejection/resume after each acceptance length from one to four tokens, including convolution history and recurrent state.
- Full GDN core with speculative decode alone, C4, one speculative request plus three prefills, and four speculative requests plus two prefills. Prefills use 330 tokens; both state dtypes pass.

Evidence: recurrent-fixed-r1.json, conv-rollback.json and mixed-core.json, with their scripts and logs. Source diffs are preserved as separate .patch files.

## Limits

Determinism here means greedy temperature-zero requests agree across the tested concurrency, cache-salt and seed settings **within this serving configuration**. The MTP-enabled configuration does not reproduce the prior no-speculation scores or output: the first returned score already differs at token zero, and the first selected-token difference is at index 38. This cross-configuration difference has not been localized; the recurrence primitive tests alone do not establish whole-model equivalence. No claim of bitwise equivalence across server configurations, restarts, stochastic sampling, or all paging/cache scenarios is made.

The existing completion-cache restriction remains: snapshots outside the 64-token grid are skipped, reducing reuse. The tests do not exhaustively validate warm multi-turn completion caching or disk-pressure paging. Maximum tested end-to-end concurrency is 8; configured capacity remains 32.

## Deployment and evidence

candidate-r0.json and candidate-r1.json are deployed. deployment-check.json verifies both live configs and all 14 source override hashes per rank. The previous stable no-spec configuration remains in ../targeted-determinism-20260913/candidate-final-r0.json and candidate-final-r1.json.

validation-summary.json records all serving assertions and timings; each probe directory retains raw SSE, token IDs, score records and usage. mixed-20260914T011021Z/metrics.jsonl records actual concurrency. metrics-final.txt contains the speculative counters captured before background trials resumed.

The active campaign on the orchestration host is 20260914T004328Z-temperature-recovery-repeat. Its supervisor and eight child request processes were temporarily suspended for isolated validation and all nine resumed after checking process start identities. The campaign contains inference-validation-pause.json and inference-validation-resume.json. Sampling configuration and retries were preserved; results spanning the restart should not be treated as a single server-configuration benchmark.
