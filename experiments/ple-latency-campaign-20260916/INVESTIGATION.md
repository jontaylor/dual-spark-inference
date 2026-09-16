# PLE latency investigation — intermediate history

**Superseded by [FINAL_REPORT.md](FINAL_REPORT.md); retained as intermediate experiment history.**

Updated 2026-09-16, 06:32 BST. This is an intermediate result, not the final serving-policy selection.

The initial in-process regression had identifiable implementation costs: every software-cache miss forced a kernel-worker handoff, even when the file page was already cached, and completion used many small polling waits. Removing those costs substantially improved the reader. Eliminating application IPC does not by itself guarantee that every new native path beats a direct cached memory load.

## Established observations

| Measurement | Before | After | Scope |
|---|---:|---:|---|
| 768-row warm-file/no-row-cache gather | 0.863 ms | 0.185 ms | Matched standalone native benchmark, forced workers vs normal issue plus batch wait |
| 1536-row spark-1 lookup, SSD power-state experiment | 1.768 ms | 0.867 ms | On/off/on returned to 1.805 ms; spark-2 unchanged comparison |
| 768-row repeated software-cache batch | 69 µs | 18 µs | Native cache/dedup experiment |
| 768-row hash/staging/publication excluding gather | 44.4 µs | 16.94 µs | GPU hash experiment |

These figures describe separate experiments and must not be multiplied or added into an end-to-end speedup. Workloads, cache conditions and measurement scopes differ.

Normal io_uring issue attempts nonblocking reads; it does not force a worker for every cached row. All unique misses are submitted before the caller waits. Full CUDA graphs now launch between native submission and completion, allowing earlier model computation to overlap the lookup. Eager execution remains synchronous to preserve correctness. Application-level PLE work stays in the GPU worker process; no PLE ZeroMQ handoff or PLE subprocess is used.

Without Nsight, the initial overlap acceptance recorded 1,072 completed GPU readiness checks with zero polls for late data, across both ranks and including 24/32-request batches. The check body took at most 0.512 µs. This does not include kernel launch/prologue and does not demonstrate GPU L2 residency.

## Remaining issue-policy tradeoff

Forced asynchronous submission gets the GPU graph launched earlier, but can finish reads later. It is under test in 64-step ABBA blocks using the same reader and cache. At 13 active requests in the latest affinity reversal capture:

| Rank | Policy | Median submission | Median GPU wait | p95 GPU wait |
|---|---|---:|---:|---:|
| 0 | Normal | 1.310 ms | 0.320 µs | 0.352 µs |
| 0 | Forced async | 0.114 ms | 228.464 µs | 973.216 µs |
| 1 | Normal | 0.941 ms | 0.320 µs | 0.352 µs |
| 1 | Forced async | 0.106 ms | 20.960 µs | 1,002.464 µs |

These are distributions, not paired per-step differences. Submission plus GPU wait is only a phase-cost proxy. Whole-step ABBA estimates have varied greatly as the representative workload changes, and do not yet establish a winner or crossover point.

The next live measurement timestamps GPU hash start and completion of the PLE readiness gate using the same GPU clock and existing kernels. It includes the earlier-launch/later-data tradeoff without relying on noisy whole-request periods. Exact CPU/CUDA graph byte checks, repeated buffer reuse, deliberately delayed publication, timeout telemetry and 300 hash differential cases passed. Candidate12 passed live mount verification and API smoke on both ranks after correcting a startup-order race. The five-minute paired capture is running; first-cycle evidence differs by rank and does not select a winner.

## Affinity and other candidates

GPU workers and their existing threads are restored to cores0–19 on both hosts. EngineCore remains15–19 as previously requested. Pinning just the GPU main thread also affected subsequently created kernel I/O workers through affinity inheritance, so that interval is not evidence for a pure caller-only effect. It did not remove asynchronous wait tails. Changing batch counts and brief compilation activity limit whole-interval comparison.

POSIX_FADV_RANDOM showed no consistent gain and is not promoted. A smaller active dedup hash produced modest isolated gains and passed native tests; integration with the experimental async API is staged, not deployed.

An exploratory first-ring SQPOLL prototype moves normal submission onto one kernel submission thread. It passes native multi-ring/byte/error/pending tests, but has a warm-read/wakeup penalty as well as shorter caller time. Initial cold larger-batch results are promising but were collected during model loading with uncontrolled CPU placement. No live deployment or GPU acceptance exists for it yet. It must be judged by exposed GPU delay rather than caller submission alone.

## Measured SQPOLL comparison

Candidate16 alternates normal submission with one kernel submission thread (SQPOLL), keeping the same in-process reader and software cache. It does not force every read onto a kernel worker. The first ten-minute live capture produced 2,636 aligned steps, no rank/policy mismatches and no native read errors.

The primary metric is GPU hash start to completion of the PLE readiness gate, measured within existing kernels on one GPU clock. Negative differences below favour SQPOLL. Cycles compare matching actual request counts and padded row counts, with at least32 samples in each of the four ABBA phases.

| Active requests / gathered rows | Complete cycles | Rank0 SQPOLL minus normal | Rank1 SQPOLL minus normal |
|---|---:|---:|---:|
| 16 / 1024 | 1 | −0.522 ms | −0.577 ms |
| 24 / 1536 | 3 | −0.741 ms | −0.830 ms |
| 25 / 1792 | 4 | −0.675 ms | −0.670 ms |

Every available complete cycle in these strata favoured SQPOLL. Cycle-bootstrap95% intervals at24 requests were [−0.815,−0.603]ms and [−0.976,−0.741]ms; at25 requests, [−0.723,−0.627]ms and [−0.716,−0.625]ms. These are small numbers of cycles under a changing representative workload, not proof of a universal optimum.

Whole-step25-request differences averaged approximately−0.67ms, but their interval spanned roughly−3.18 to+3.58ms. Therefore this establishes an improvement in the measured PLE readiness path, not a statistically resolved end-to-end speedup. The gate does not certify that all embedding bytes already reside in GPU L2.

The longer same-process capture is continuing to cover smaller request counts. A fixed-policy candidate17 is staged but not GPU-validated or deployed. It makes SQPOLL opt-in, creates no polling ring for the normal path, and rejects accidental experimental switching. Serving still runs candidate16 ABBA; a final fixed policy has not yet been selected.

Evidence: `results/ple-latency-campaign-20260916/sqpoll-ab-first/{gate-comparison,step-comparison,summary}.json`. The earlier candidate12 used forced IOSQE_ASYNC, a different policy; do not pool its counters or results with16.

## Operational state and limits

The NVMe latency policy is now persisted by verified udev rules on both nodes; it increases idle SSD power. Device-event tests passed, but reboot persistence has not been tested by rebooting.

Current exact variant, rollback paths and acceptance status are in WORKLOG.md. Frozen assets remain separate from the canonical development checkout. API smoke is a functional check, not broad model-quality certification; prior supplementary verification failures remain separate from this performance investigation.

Evidence is under ../../results/ple-latency-campaign-20260916/, including async-ab-main-restored/comparison.json, gate-timing-integration.log, random-advice-benchmark.json, sqpoll-benchmark.json and sqpoll-idle-benchmark.json.
