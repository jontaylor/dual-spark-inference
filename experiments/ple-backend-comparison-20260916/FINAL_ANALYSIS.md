**The completed comparison does not demonstrate a throughput winner.**
The original external PLE workers delivered **261.46 output tokens/s** and the
in-process backend **262.57 output tokens/s** on the fixed 12-request workload.
The paired geometric difference favors in-process by **0.42%**, with a
preliminary 95% interval of **−0.78% to +1.65%**. This is evidence against a large
gain on this workload, not proof of equivalence or a ranking for other workloads.

At campaign completion, in-process was selected on both nodes (epoch 14). The API was healthy, the
repeating client is held idle, and neither container nor its worker processes
restarted during the comparison. Both complete backends are now available in
one deployment; that deployment allowed subsequent comparisons without a model reload.

The user subsequently selected original external workers for maintainability.
See [the external-only restoration](../ple-external-restore-20260916/README.md).
This operational choice does not change the measured result below. Compact
publication evidence is retained in [evidence/](evidence/); large raw captures
remain in the local results directory.

**Measurement and repeatability.** Two warmups preceded eight measured cycles,
with backend order external/native/native/external/native/external/external/native.
Each cycle ran the same 12 distinct prompts concurrently, generating exactly
2,048 tokens per request. Temperature was 0, top_p 0.95, top_k 20, with medium
reasoning. Both backends used s32/b8192/t1024, TP2/MTP3 and unrestricted CPU
affinity. Each cycle drained before the next verified two-node backend switch.

| Measured cycle | Backend | Elapsed seconds | Output tokens/s |
|---|---|---:|---:|
| 1 | External | 95.12 | 258.37 |
| 2 | In-process | 93.77 | 262.08 |
| 3 | In-process | 93.39 | 263.15 |
| 4 | External | 93.86 | 261.85 |
| 5 | In-process | 93.72 | 262.24 |
| 6 | External | 93.37 | 263.22 |
| 7 | External | 93.64 | 262.45 |
| 8 | In-process | 93.50 | 262.83 |

Each backend generated 98,304 useful output tokens over four cycles. The
headline rates divide those tokens by total cycle duration, including prefill
and the final concurrency drain. Adjacent opposite-backend pairs favored
in-process by +1.44%, +0.50%, −0.37% and +0.15%. The interval uses a Student-t
estimate on those four log ratios (three degrees of freedom), conditional on
repeatability and independence; individual tokens are not independent trials.
The first external cycle was the slowest, and the later comparisons were closer.

| Client measurement | External | In-process |
|---|---:|---:|
| Mean time to first token | 13.828 s | 13.817 s |
| Mean time per output token | 38.450 ms | 38.264 ms |
| Measured requests | 48 | 48 |
| Uncached prompt tokens per cycle | 42,302 | 42,302 |

Client TPOT divides generation span by output tokens minus one; it is not the
distribution of individual token delivery gaps. Server counters independently
confirmed exactly 12 completions, 24,576 generated tokens, 180,542 total prompt
tokens and 138,240 cached prompt tokens in every measured cycle. Running and
waiting gauges were zero at each recorded boundary. No extra completions or
client errors were recorded.

The independent client-thread audit covered all 120 requests including warmups:
108 per-lane repeat comparisons, zero payload mismatches, zero generated
content/reasoning/finish-reason mismatches and zero output-token-count mismatches.
This establishes repeatability of these outputs, not general model quality or
token-logprob identity.

**What this says about the critical path.** In full decode at 12 requests, the
mean measured GPU iteration was approximately 142.29 ms for external and
141.55 ms for in-process on each rank. These are descriptive groups, not an
independent causal estimate after exact context/speculation matching. Draft
acceptance was approximately 89.84% and 89.83%, respectively.

PLE readiness waits were very small in both implementations:

| Full-decode wait-loop measurement | External rank 0 | External rank 1 | In-process rank 0 | In-process rank 1 |
|---|---:|---:|---:|---:|
| Mean body duration | 0.365 µs | 1.150 µs | 0.345 µs | 0.366 µs |
| Steps with a wait-loop poll | 1 / 2268 | 7 / 2268 | 0 / 2267 | 0 / 2267 |
| Maximum body duration | 10.176 µs | 629.024 µs | 3.744 µs | 8.224 µs |

These include the small ready-check cost and exclude kernel launch/prologue.
The external rank-1 tails occurred below 12 active requests. They were rare and
small relative to the roughly 140 ms iteration. The consumer usually reached
PLE after the data was ready with either backend. Removing the message handoff
therefore does not automatically create a meaningful serving-throughput gain.

This wait metric does not include input staging, hashing, host synchronization,
GPU memory access/dequantization/projections or resource contention. In prefill,
the measured prepare call averaged roughly 65–69 ms per rank for both paths,
within approximately 1.98-second measured GPU iterations. That call includes
synchronization and cannot be interpreted as pure storage latency or fully
recoverable overhead. There were 28 prefill-containing steps per backend and
no non-FULL decode-only steps in the measured data. This run does not settle
performance for those other execution shapes or cold page faults.

**Operational outcome and limits.** Keep the present in-process selection; the
data does not justify declaring it faster or switching back for a speed gain.
The useful next investigation is whether real mixed/prefill or memory-pressure
traffic exposes PLE startup or readiness delay. Scheduler readiness gating is
worth pursuing only where late data is actually holding up useful work; warm
decode here gives little evidence of that opportunity. The broader options and
dependency analysis remain in [the critical-path review](../ple-critical-path-20260916/ANALYSIS.md).

Both original software caches remain resident in the comparison deployment and
share the OS file cache. Repeated identical prompts/outputs make this a warm
workload; cold I/O, other concurrency levels, single-backend memory pressure
and agent quality remain outside this result. Both arms had the same event/log
instrumentation, whose overhead was not separately quantified. Neither arm
performed shadow PLE lookups.

The telemetry audit found 5,851 aligned completed steps across the whole
deployment, including canaries and warmups, with zero reported errors and
contiguous sequences on both ranks. Final sequence 5852 has a begin record on
both ranks; its completed timing is buffered until another request or clean
close. Headline client throughput includes its output. No further request was
introduced merely to flush it.

Frozen source hashes, runtime mounts, serving parameters, both control files
and unrestricted thread affinities passed final verification. HTTP health was
200. Backend control is fixed in_process, epoch 14; the separate reader-policy
control stays zero (SQPOLL), with rejected prefetch disabled. The former
native-only uprobe monitor stays stopped because it cannot attribute backend
switches correctly. Common per-step JSONL telemetry remains active on both ranks.

Reproduce the offline analysis from the collected campaign evidence:

```
.venv/bin/python experiments/ple-backend-comparison-20260916/analyze.py
.venv/bin/python experiments/ple-backend-comparison-20260916/review.py
```

Evidence is in `results/ple-backend-comparison-20260916/`: `comparison.json`,
`final-summary.json`, `repeat-identity-audit.json`, per-cycle Prometheus snapshots,
`final-health.json`, `final-backend.json`, `live-verification.json`, raw client
records and both ranks' step logs. Backend control and safe repeat procedures
are documented in [README.md](README.md). Do not edit the loaded frozen files
or rerun the completed campaign against its existing result directory.
