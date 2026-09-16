# Memory completion cache rollout — 2026-09-13

Both TP ranks now run source revision
`318e06e84d2aeb70a240531705ae67fa3f977b7a`, with
`disk_write_policy=pressure`, `memory_completion_cache=true`, and
`save_timeout_seconds=300`. All 11 deployment manifest hashes were verified.

The earlier pressure-only change stopped routine disk saves but also stopped
saving exact completion checkpoints. Follow-ups could therefore lose reusable
state between aligned checkpoints. Completed-request snapshots now stay in the
existing GPU KV block pool and count against its physical and reservation budgets.
Admission and growth spill idle snapshots to disk when their memory is needed;
pages remain pinned until both ranks acknowledge the write. Completion saves
fall back to disk if a resident snapshot cannot fit. Memory-only transfers report
zero disk bytes. Existing model, PLE settings, 40 GiB KV and 48 GiB disk budgets
per rank are preserved. Snapshot slots share the existing logical slot limit.

## Validation

The focused suite passed 56 tests on each node's GPU (52 CPU tests plus four GPU
checks), including memory-only and mixed memory/disk round trips, checksums,
capacity accounting, delayed reuse, and two-rank spill acknowledgement. Source
pre-commit/type/lint checks passed before freezing the committed revision.

Controlled synthetic follow-ups generated 32 tokens, then submitted the emitted
history plus five new input tokens. The final emitted token requires replay.

| Initial prompt tokens | Prior cached tokens | New cached tokens | Prior follow-up seconds | New follow-up seconds |
| --- | --- | --- | --- | --- |
| 1,599 | 0 | 1,630 | 0.983 | 0.785 |
| 6,000 | 3,200 | 6,031 | 1.355 | 0.789 |
| 11,000 | 8,000 | 11,031 | 1.669 | 1.102 |

An aligned 1,569-token prompt also reused all 1,600 processed history tokens.
All four low-pressure cases produced zero disk store and load bytes. These are
small synthetic latency samples, not a representative decode throughput result
or a production cache-hit-ratio measurement.

Pressure preparation retained 700 resident snapshot pages without disk writes.
A subsequent batch of 32 concurrent requests triggered 7,105,945,600 aggregate
disk-store bytes, completed all requests, and incurred zero preemptions. A 100K
follow-up reused 100,015 tokens. That follow-up and the first repeated pressure
probe still read resident pages (zero disk-load bytes), so they do not establish
model-level disk restore.

The final targeted run used six waves of 32 concurrent requests without touching
the selected snapshot between waves. All 192 requests succeeded with zero
preemptions. The follow-up then read 316,825,600 aggregate bytes from disk, reused
11,031 of 11,037 input tokens, and exactly matched all 32 tokens from the earlier
resident replay. No spills remained pending. This establishes actual disk restore
for the tested snapshot; evidence is in `spilled-restore-final.json`.

## Numerical scope

Three initial cached cases matched all 32 greedy cold-control tokens. The 11K
case differed from three mutually identical cold controls starting at token two.
The first-token log probability differed by approximately 0.011. This is a real
execution-path difference; it is not evidence of bit-exact cold equivalence.

A separate control generated 64 uninterrupted tokens from the original 11K
prompt. Its first 32 tokens matched the original source generation, and its last
32 exactly matched the cached continuation of that source generation. This
supports decoded-state continuity for the tested case, but does not certify
numerical equivalence when arbitrary new input is appended. Earlier disk-only
completion checkpoint validation also recorded cold-versus-resume differences;
see `completion-checkpoints.md`.

## Evidence and operation

Private scripts and JSON responses are under
`/home/jon/memory-completion-rollout-20260913/`: `baseline.json`, `after.json`,
`numeric-check.json`, `continuity-check.json`, `pressure-check.json`, and
`spilled-restore-final.json`.
The same directory on each node holds its `rankN-tests.log` and original
`rankN-before.tar.gz` rollback archive. The candidate bundle is retained there.

Synthetic snapshots remain in the bounded, evictable cache after validation.
Their presence can temporarily affect aggregate hit metrics and pressure. No
additional restart was performed just to clear them.

To restore the prior pressure-only deployment, stop the head service
`qwen38-next-qwen-fp8.service`, extract each node's corresponding
`rankN-before.tar.gz` into `/home/jon/dual-spark-inference-kv-paging`, and restart
the head service. It coordinates the worker. These archives restore the outgoing
configuration, launcher, manifest and overlays, and must not be overwritten.
