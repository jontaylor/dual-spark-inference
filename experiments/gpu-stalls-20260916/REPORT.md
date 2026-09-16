# Live GPU utilisation investigation — 16 September 2026

No serving changes, restarts, affinity changes or test requests were made.
Original external PLE workers remained deployed. Captures and profilers have
finished on both hosts.

## Findings

Two simultaneous two-rank captures covered 16:39:49–16:43:19 UTC and approximately
16:45:16–16:50:16 UTC (UK time is UTC+1). NVIDIA utilisation was sampled every
200 ms. Values repeat because the underlying sensor averages over a longer
interval; these are not independent 200 ms GPU activity measurements.

| Capture | Rank | Mean GPU utilisation | Minimum |
| --- | --- | ---: | ---: |
| First, 210 seconds | 0 | 95.31% | 78% |
| First | 1 | 94.75% | 55% |
| Follow-up, 300 seconds | 0 | 95.18% | 75% |
| Follow-up | 1 | 94.81% | 86% |

We reproduced brief asymmetric dips, but not a deep or sustained GPU outage.
The captures do not establish an exact causal stack for a dip.

### User-supplied times

- 17:39:55 UK: rank 1 reported 81% from about 16:39:55.346 UTC;
  rank 0 remained 94–96% in the surrounding ten seconds. PLE tracing had not
  started yet (the first capture included a 40-second unprofiled baseline).
- 17:41:16 UK: both ranks reported 95–96% at that time. The nearby rank-1 dip
  was 63%, at 16:41:11.443–16:41:12.244 UTC. Rank 0 stayed around 95–96%.
- 17:35:43 UK predates these captures and cannot be attributed from them.

The 63% episode overlaps a decode-to-mixed-prefill transition: 8 requests / 32
padded tokens became 9 requests / 1,056 tokens, followed by 928 tokens. Step
submission intervals increased from approximately 130 ms to 496 and 433 ms.
PLE was ready in approximately 10–17 ms during those larger steps. This is a
workload transition, not evidence that PLE consumed the rest of the interval.
Longer prefill steps also perform substantially more GPU work.

The other marked rank-1 dip, 55% at 16:40:53.423–16:40:54.224, also coincided
with a request/prefill transition. Its nearby PLE readiness maximum was 30 ms.
Five follow-up rank-0 dips reached 75–81%, with nearby PLE readiness maxima of
26–36 ms. The follow-up had no utilisation below the 60% stack-dump trigger.

### PLE

| Capture / rank | Gather count | Median gather | Max gather | Median request-to-ready | Max request-to-ready |
| --- | ---: | ---: | ---: | ---: | ---: |
| First / 0 | 743 | 2.610 ms | 40.927 ms | 3.920 ms | 42.137 ms |
| First / 1 | 743 | 2.463 ms | 34.484 ms | 3.622 ms | 45.476 ms |
| Follow-up / 0 | 2,582 | 1.547 ms | 30.685 ms | 2.722 ms | 35.587 ms |
| Follow-up / 1 | 2,583 | 1.609 ms | 14.493 ms | 2.787 ms | 37.299 ms |

These are CPU publication-to-completion times, not exposed GPU wait or proof
of GPU cache residency. PLE main threads mostly waited for incoming work.
There is no observed hundreds-of-milliseconds PLE service stall in the traced
windows. This does not prove PLE has zero performance cost.

### CPU and paging

In the follow-up, main worker threads spent 26.3 / 258.5 sampled seconds on
rank 0 and 22.8 / 260.2 on rank 1 at `short_conv_attn.py:333`: the synchronous
`spec_req_idx_cpu.to(query_start_loc.device)` during attention metadata
construction. This is a concrete candidate for replacing per-step blocking
copies with reusable pinned buffers and correctly ordered asynchronous copies.
Time there can include waiting for earlier CUDA work; it is not an estimate
of recoverable speedup and does not establish the cause of an individual dip.

Most main-thread samples were output-event synchronization while awaiting
GPU results (203.9 and 201.2 seconds). Such stacks alone do not mean the GPU
was idle. The PLE input-copy synchronization accounted for about 1.1 sampled
seconds per rank. CPU run-queue delay for each worker's main thread was under
10 ms across each entire capture, so CPU scheduling starvation is not supported.

Disk stores, restores and preservation occurred during the captures. Current
eviction code preserves asynchronously; the September 14 full-persistence-fence
diagnosis must not be applied to this deployment. The shared disk executor and
finite staging slots remain possible sources of backpressure, but the profiles
do not establish that they caused these dips. Store log durations can include
queue time and are not equivalent to inference stalls.

The actual server running-request gauge ranged from 6–10 in the first capture
and 2–8 in the follow-up. Client agent/request counts are not the same as the
number of requests participating in GPU batches. The finite workload was
changing as arms completed and agents entered tool/planner phases.

## Limits and next measurement

Python profiles were nonblocking and include idle threads. Missing/error samples
prevent exact timestamp reconstruction from profile weights. The first rank-0
worker profile failed to save; follow-up profiles saved on both ranks. Invalid
UTF-8 in nonblocking profile output was decoded with replacement for analysis.
The cross-host wall-clock offset was measured at approximately 1.935 ms
(remote ahead), far too small to explain seconds of apparent offset.

NVIDIA utilisation is not a CUDA critical-path trace. It cannot distinguish
host launch gaps, rank-local copies, dependencies, and execution imbalance.
The next diagnostic should capture timestamped stacks at a higher dip threshold
and, if ambiguity remains, a bounded CUDA timeline of copies, kernels and waits.
Only that timeline can establish the idle interval and its blocking dependency.
Do not restart or redesign PLE based on these utilisation readings alone.

## Evidence

Raw files: `results/gpu-stalls-20260916/rank{0,1}/` and
`results/gpu-stalls-20260916/extended/rank{0,1}/` (ignored by Git).
Each contains timestamped GPU readings, process/host counters, Docker logs,
PLE uprobes, profile start metadata and available Speedscope profiles. Rank 0
also contains Prometheus snapshots. `capture_host.py` records these inputs.
`clock-check-persistent.json` records the cross-host clock check.
