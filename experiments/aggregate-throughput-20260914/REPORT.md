# Completion cache reuse repair — 14 September 2026

The deterministic completion-checkpoint patch introduced a cache regression: lookup required a 64-token boundary, but completion still stored arbitrary boundaries. In a 55-minute sample, 339 of 343 saves were unaligned; 9,348 of 9,477 saved page references belonged to unusable checkpoints. These checkpoints consumed resources but could not satisfy the aligned lookup. This is a defect in our patch, separate from the earlier upstream batch-dependent numerical behavior.

Completed-request evidence agrees with the user's cache decline: the earlier temperature sweep reused 18,782,228 of 19,702,263 prompt tokens (95.33%, 580 requests); the later MTP epoch reused 8,782,400 of 11,229,213 (78.21%, 243 requests), recomputing 2,446,813 tokens versus 920,035 previously. These runs have different workload outcomes and are not a controlled throughput comparison. See `run-comparison.json`.

## Running repair

V2 retains an immutable CPU snapshot of the latest available accepted recurrent and circular-buffer state at a 64-token boundary, then uses that snapshot at completion. Full attention prefix pages are copied only on completion. The reusable boundary is rounded down, leaving at most 63 tokens between that boundary and the computed end. This bound does not cover new prompt/tool text appended afterward. Unavailable historical states are rejected, never advertised as valid hits.

MTP3, CUDA graphs, concurrent batching and scoped deterministic kernels remain enabled. The scheduler is byte-for-byte identical to the previously validated MTP scheduler. BF16 projections still use batched fixed-reduction GEMMs. There is no request serialization or per-request GEMV in this repair. Both live deployment configurations and all 15 runtime override hashes per rank were verified (`deployed-verification-v2.json`). Configuration: `candidate-aligned-r0.json` / `candidate-aligned-r1.json`; generator: `build_aligned.py`.

The first candidate clipped speculative steps at checkpoint boundaries. Mixed-C8 testing caught token/score divergence exactly after those boundaries; that candidate was rejected. V2 removes clipping and captures retained speculative states after crossing the boundary. `scheduler-checks.json` documents rejected V1, not the final policy.

## Controlled aggregate throughput

Same running V2 server, identical 7,393-token input, 16 generated tokens per request, four concurrent requests. Alternating partial/full/full/partial phases; independent salts prevent the first partial phase from warming the second. Shorter-prefix seeding occurs outside timed phases. All phases started idle, all completed successfully, and actual cache counts were verified.

| Reuse condition | Cached tokens/request | Recomputed tokens/C4 | C4 wall time | Aggregate generated tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Partial prefix A | 4,800 | 10,372 | 4.016 s | 15.94 |
| Repaired completion A | 7,360 | 132 | 1.919 s | 33.35 |
| Repaired completion B | 7,360 | 132 | 1.836 s | 34.86 |
| Partial prefix B | 4,800 | 10,372 | 3.995 s | 16.02 |

Across the two repetitions per condition, aggregate generated throughput improves approximately 2.13×; recomputed prompt work drops 98.73%. This measures the benefit of additional reuse on the same server, including checkpoint overhead. It is not a claim of a 2.13× improvement for every workload. Different cache routes can produce different output text; generation lengths are fixed. Raw responses and timings: `reuse-benchmark-v2/`.

The direct continuation probe also now caches 7,360/7,393 tokens (99.55%), versus 4,800 before. The original before/after latency observations had different background loads, so the controlled benchmark above is the performance evidence.

## Correctness validation and limits

- CPU tensor checks cover accepted speculative state normalization, crossing a boundary by 1–3 tokens, circular-buffer retention, immutable snapshots, cleanup, metadata serialization and unavailable-state rejection. See `checkpoint_checks.py` and `checkpoint_crossing_checks.py`.
- Original full-response serial/concurrent/serial probes pass exact prompt IDs, generated IDs and score records: 10 requests without a fixed seed, 10 with a fixed seed (`model-validation.json` and `../spec-determinism-20260914/full-aligned-v2*`). Each completed response has 125 generated tokens.
- Mixed short/long-prompt test passes exact tokens and scores: 16 responses, including eight staggered concurrent requests. The harness now asserts all equality results.
- Warm continuation C1/C4 passes exact tokens/scores; independent zero-cache C1/C4 also passes.
- Independent state oracle: a request finishing exactly at boundary 7,360 uses the original current-state capture. Its restored continuation exactly matches tokens and complete score records from the retained historical snapshot. This validates the new snapshot route against an independent capture route (`continuation-equivalence-v2/summary.json`).
- **Cache-history-independent determinism remains unresolved.** Restored continuation differs from uninterrupted generation, and cached replay differs from fully cold prefill. Different prefill/recurrent arithmetic is a suspected explanation, not established by these tests. Passing the state oracle does not prove cold/warm equivalence. Cross-restart/config equality is also not established; this boot's full response differs from the earlier 145-token MTP reference.
- CPU snapshot memory is bounded by active request slots and overwritten at each new boundary. Capture introduces device-to-host copies/synchronization approximately once per 64 generated tokens; its isolated cost has not been profiled. The controlled benchmark includes it.

## Workload restoration

All seven recorded campaign processes were identity-checked and resumed, children before supervisor, at 02:38:29 UTC. The current owner task was notified; workload sampling, cadence and retries were retained. See `campaign-resume-v2.json`. The campaign spans configurations and deliberate validation pauses, so whole-campaign elapsed time cannot be treated as a single-configuration throughput measurement.

Post-resume interval observations are recorded in `post-resume-metrics.jsonl`, measured against `pre-resume-metrics.txt`. Server cumulative counters include validation and cold startup; they should not be compared directly with the previous warm campaign's lifetime cache ratio.

## Observed real workload after resume

Through the final response snapshot, 27 completed requests reused 1,322,944/1,633,987 prompt tokens (80.96%), including six initial requests rebuilding cache after restart. The subsequent 21 completed continuations reused 1,318,848/1,327,198 (99.37%); only 8,350 prompt tokens were recomputed. This subset is defined by excluding the first completed request per workload arm, not by selecting high-hit responses. Raw usage and classification: `post-resume-responses.json`, `post-resume-response-summary.json`.

Server counters over the full 329.4 seconds since resume report 83.21% prompt reuse and 63.4 aggregate generated tokens/s, including startup cache rebuilding. Over seconds 139.2–329.4, after initial prefill, reuse is 99.30% and aggregate generation 101.7 tokens/s (190.2 seconds, 24 completions). These server metrics include in-progress generation and account prompt work at different times from completed-response usage; their denominators therefore differ. This is a short live observation, not proof of a sustained steady-state rate or a matched historical throughput comparison.

The captured live checkpoint log contains 25 saves, all 64-aligned, with no skipped checkpoint or ERROR entries in that sample. API health remains successful and six requests were running at the final metric sample. `post-resume-checkpoints.log` and `post-resume-interval-summary.json` preserve evidence. The repaired configuration remains running.
