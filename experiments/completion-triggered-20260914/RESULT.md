# Running result: MTP3, completion/suspend-triggered checkpoints

The running service uses MTP3 and no longer requests or captures a checkpoint shadow at every 64-token decode boundary. Completion and suspend saves retain the exact accepted state. An unaligned restore processes a bounded initial partial chunk before returning to the absolute 64-token arithmetic grid. Concurrent batched GEMMs and scoped deterministic kernels remain enabled. Full checksums/GPU readback remain enabled.

The unnecessary MTP3-only baseline load was stopped on the user's instruction; the next completed load was this candidate. No extra baseline run was performed.

## Validation

- Both nodes' configurations and all 18 source overrides verified; API healthy, containers running without OOM.
- CPU accepted-state/layout gates: 40 cases. Partial-grid scheduling: 1260 cases, covering all 63 offsets and insufficient batch budgets. Ordinary decode capture is a no-op.
- Seven live completion/follow-up cases, including natural-language prompts, length stops and a token-triggered stop. All 28 C4 comparisons matched generated token IDs and returned logprobs exactly. Every follow-up reused the expected exact checkpoint boundary. No broad cold-versus-cached equality claim.
- Four waves of 32 submitted pressure requests completed; the chosen continuation retained exact tokens/scores and its expected cache hit. No preemptions reported in the pressure snapshots. Actual concurrency is scheduler-dependent; this is not a C32 execution claim.
- A separate attempt to compare a chosen disk checkpoint with an independent resident checkpoint did not establish unambiguous foreground disk coverage: targets stayed resident or other restores overlapped. Preserve this coverage failure; do not report a disk-oracle pass. Existing transport verification and suspend-save logic are retained. A new forced suspension cycle was not exercised.

## Performance

Head-worker main-thread sampling attributed 1.55% to event checkpoint capture, versus 10.3% in the earlier periodic-capture profile. PLE preparation fell from 15.9% to 0.29% in these samples. GPU-output waits rose to 76.8%. These are stack-residency observations under different workloads, not isolated removable-cost percentages.

A three-minute mixed warm-workload sample averaged 107.68 aggregate tokens/s with 5.08 mean sampled running requests and 85.10 W mean sampled combined GPU power. It includes probes and workload transitions; it is not comparable directly to the earlier approximately nine-request historical window. The separately preserved cold-start sample averaged 17.85 tokens/s while long prompts were recomputed after restart.

Capacity checks with normal background requests continuing:

| Check | Cache verification | Elapsed | Probe tokens/s | Whole-server aggregate tokens/s |
| --- | --- | ---: | ---: | ---: |
| Eight identical 8,005-token follow-ups | 8/8 reused 8,003 tokens | 36.25 s | 113.01 | 153.67 |
| Eight diverse technical follow-ups | 8/8 reused exact completion boundary | 30.41 s | 134.70 | 177.58 |

The diverse check produced a ten-second server generation window of 202.1 tokens/s with 10 running requests. This demonstrates usable concurrent capacity, not restoration of the entire historical workload's throughput. The initial attempted warm C8 check had mostly cold cache hits after pressure; its scope was corrected and it is excluded as a warm-decode comparison.

## Configuration

MTP3, TP2/expert parallel, BF16 KV, 40 GiB GPU KV and 48 GiB logical disk per rank, 32 sequences, 8192 batched tokens, 1920-token retention, 7680-token reservation increments. Event source files and candidate-events-r0/r1.json are in this directory. FINAL-AUDIT.json records actual live identities and hashes. No diagnostic override was left enabled.
