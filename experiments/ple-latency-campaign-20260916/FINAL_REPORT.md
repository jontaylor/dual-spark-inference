# PLE latency investigation — 16 September 2026

Final candidate: in-process reader17 with fixed SQPOLL submission, connector17,
GPU hash/wait12 and overlap runner06. Deployed on both ranks and accepted after
API smoke, configuration verification and a full 20-minute stability capture.

Subsequent change: EngineCore was unpinned to CPUs 0–19 for all threads at the
user’s request later on 16 September. The affinity settings below describe the
original campaign window. See `results/ple-engine-unpin-20260916/REPORT.md`
for the subsequent short comparison; no clear matched-batch speedup was found.

## What caused the apparent crossover

Moving PLE inside the GPU process removed the application IPC handoff, but the
first implementation added avoidable kernel scheduling. It forced every
software-cache miss through a kernel worker, even when the file data was already
cached, and repeatedly polled completion. A live 768-row batch had median native
latency 3.608 ms, including 3.125 ms accumulated ppoll time and 44.5 polls per batch.

Removing forced worker dispatch and using a batch completion wait reduced a
matched warm-file, software-cache-disabled 768-row microbenchmark from 0.863 to
0.185 ms. The legacy 16-thread direct gather took 0.374 ms before its IPC costs.
However, a legacy single-thread mmap gather over cached memory took only
0.0119 ms. An I/O API cannot be assumed to beat direct cached memory loads in
every case. These microbenchmarks are not live end-to-end comparisons.

NVMe power-state exit was another substantial source of latency. A causal
spark-1 on/off/on test at 1,536 rows measured 1.768 → 0.867 → 1.805 ms while
spark-2 stayed unchanged. With the latency constraint applied on both nodes,
the corresponding measurements were approximately 0.855/0.884 ms.

## Implemented changes

- Keep the reader, bounded row cache and result ownership inside the GPU worker;
  no PLE subprocess or per-step ZeroMQ handoff.
- Submit every unique missing row before the first completion wait. A separate
  cold-advised 9,000-row test submitted 4,096 + 4,096 + 808 reads before its one
  completion wait and verified exact output bytes.
- Remove unconditional IOSQE_ASYNC and repeated userspace completion polling.
- Copy row-cache hits directly into output and deduplicate with an epoch hash.
  Repeated-cache 768-row gather fell from approximately 69 to 18 µs.
- Compute exact row IDs on the GPU. Staging/hash/publication overhead excluding
  gather fell from 44.4 to 16.94 µs at 768 rows in the isolated comparison.
- For real FULL CUDA graphs, submit reads, launch independent early graph work,
  then complete and publish the host result. Eager/piecewise execution remains
  synchronous to avoid a publication deadlock.
- Use one optional SQPOLL submission ring for the first 4,096 unique misses;
  larger batches submit remaining rings normally, still before any wait.
  This is a kernel submission thread, not a PLE application worker/message bus.
  Polling idle timeout is 10 ms. Normal-policy readers create no SQPOLL ring.
- Apply NVMe latency tolerance 0 on both nodes, persisted with
  `/etc/udev/rules.d/99-gb10-ple-nvme-latency.rules`.

The GPU readiness check establishes that the complete host result is available
with the required memory ordering. It does not establish that all embedding
bytes have reached GPU L2. The measured GPU interval below starts at row hashing
and ends at this readiness check; it includes intervening graph work.

## Controlled live policy comparison

Reader16 alternated ordinary nonblocking issue and SQPOLL in 64-step ABBA blocks
using the same reader/cache. The comparison matches exact row count and actual
request count, excludes switching boundaries, requires at least 32 samples in
every phase, and bootstraps whole complete cycles. Negative differences favour
SQPOLL. The old experiment label `async` means SQPOLL here, not IOSQE_ASYNC.

| Actual requests | Rows | Complete cycles | Rank 0 gate difference, ms (95% interval) | Rank 1 gate difference, ms (95% interval) |
|---:|---:|---:|---:|---:|
| 12 | 768 | 14 | −0.244 [−0.274, −0.214] | −0.224 [−0.262, −0.184] |
| 16 | 1,024 | 1 | −0.522 (one cycle) | −0.577 (one cycle) |
| 24 | 1,536 | 6 | −0.754 [−0.834, −0.675] | −0.810 [−0.905, −0.717] |
| 25 | 1,792 | 6 | −0.802 [−0.965, −0.664] | −0.797 [−0.952, −0.659] |

Every complete cycle favoured SQPOLL on both ranks. There was no observed reversal
at 12 requests in this comparison. This compares two improved in-process
submission policies, not the entire original out-of-process implementation.

At 12 requests the gate medians were 2.511/2.511 ms with SQPOLL versus
2.750/2.717 ms with ordinary submission; p95 values were 2.846/2.785 versus
3.242/3.157 ms. To obtain a filled 12-request comparison, the server sequence cap
was temporarily set to 12; client sampling/concurrency was unchanged. The final
configuration restores the cap to 32. Its 3,962 matched sequences have zero rank
shape/policy mismatches and zero native errors. The separate larger-batch
captures contain 5,462 matched sequences with zero mismatches/native errors.
Sequence counters were not pooled across restarts.

Whole-step latency remains statistically unresolved: at 12 requests the paired
mean change was −0.418 ms, with 95% interval [−1.315, +0.510] ms. The initial
25-request capture was also inconclusive. These are proven readiness-path gains,
not a claim of equivalent whole-request speedup or a universal optimal policy.
Dynamic workload differences still limit causal attribution.

Evidence: `results/ple-latency-campaign-20260916/sqpoll-ab-s12/` contains
`gate-comparison.json`, `step-comparison.json` and both rank CSVs;
`sqpoll-ab-combined-gate.json` contains the separate larger-batch comparison.
The second larger-batch capture lasted 573.115 seconds and ended cleanly;
it was not a completed 1,200-second capture.

## Validation and deployed settings

Both fixed normal and fixed SQPOLL policies passed GPU integration before
promotion: CPU/CUDA exact output bytes, real graph replay, 20 buffer reuses,
delayed publication, timeout telemetry, and 300 differential hash cases.
Native tests covered collision/output lifetime, duplicate rows, failure paths,
optional SQPOLL setup and pending-batch rejection. An additional 135 policy
switches covered 64–9,000 rows and three cache sizes. Canonical native pytest:
7 passed; Ruff and diff whitespace checks passed. Canonical source hashes match
the GPU-validated sources (`final-source-sha256.json`).

Final serving settings: s32/b8192/t1024, TP2, MTP3, fixed SQPOLL, no ABBA
switching, no forced IOSQE_ASYNC. GPU workers and their threads are unrestricted
on CPUs 0–19; EngineCore is pinned to 15–19 per the user's current setting.
No pure affinity benefit was established. The NVMe rule passed udev validation
and actual device-event testing; a reboot was not tested. Keeping the SSD awake
can increase idle power.

Functional API smoke and exact-byte tests do not certify model quality.
Pre-existing supplementary verification failures remain a separate concern.

## Final live acceptance

Completed at 07:35 BST. API smoke passed authentication, both served aliases,
concurrent arithmetic, tool calls and streaming. Both launcher dry runs rehashed
the configured assets successfully; live mounts, environment and actual serving
arguments were verified again after capture. Canonical source hashes still match
the validated sources.

The final fixed-policy capture lasted **1203.404 seconds**:
5,880 completed batches on rank 0 and 5,890 on rank 1, with **5,880 common sequences,
zero rank shape/policy mismatches and zero native errors**. Rank1's ten additional
sequences are all after rank 0's capture boundary, not missing middle batches.
The full trace audit passed. Both ranks used synchronous gather for 733 eager
batches and SQPOLL for all deferred FULL graph batches (5,147/5,157 respectively).
Capture stderr was empty; service logs contained no errors, tracebacks or timeouts.
Worker PIDs 127401/3265155 were unchanged, both containers had zero restarts/OOM
kills, and the health endpoint returned 200 after capture.

Across this mixed-batch stability workload, GPU readiness-check body p95 was
0.384/0.512µs, with median 0.352µs on both ranks. Rare late-data waits remain:
maximum 3.269/1.748ms. Hash-to-ready-gate medians were 3.660/3.648ms, p95
4.662/4.729ms, and maxima47.027/52.506ms. Those broader intervals include
intervening GPU work and scheduling. They are not a matched before/after
comparison and do not demonstrate that all latency tails have disappeared.

Evidence: `final-stability/audit.json`, `summary.json`, both raw rank traces and
CSVs; `final-smoke.json`, `final-live-processes.json`, `final-service-state.json`
and `final-worker-affinity.json`, all under the campaign results directory.
The unchanged representative client campaign was
`20260916T061335Z-restart-reset-013-restricted-temp0`, with 16 conditions,
temperature 0 and medium reasoning (`final-load-settings.json`). Load remains
running; no additional restart is planned.

## Rollback

The GPU-validated fixed ordinary-submission rollback is
`deploy.fixed-normal-r0.json` / `deploy.fixed-normal-r1.json` in this directory.
Stop serving, install BOTH rank configs on their respective hosts as
`deploy_config.json`, then start the head. The head's ExecStartPre starts the
remote worker, so installing the remote config after head start is too late.
Verify both live mounts/env/serving parameters with `verify_live.py`, run the API
smoke, and restore EngineCore affinity only after the GPU worker has spawned.
Do not overwrite mounted frozen assets. No systemd daemon-reload is required.

To roll back the NVMe policy, remove the above rule on both nodes, reload udev
rules and restore `/sys/class/nvme/nvme0/power/pm_qos_latency_tolerance_us` to
100000. This is independent of the submission-policy rollback.

The campaign retains rejected variants and raw evidence in WORKLOG.md and
`results/ple-latency-campaign-20260916/`. No commits or pushes were made.
