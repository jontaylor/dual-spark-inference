# Serving result: GPU completion cache, MTP3

The candidate-v2 configuration is deployed and left running. Runtime hashes on both ranks match all 22 overrides (runtime-verified-v2.json). No additional baseline reload was performed.

## Change and cause

Removing completion reuse exposed native matching on coarse 1920-token pages, with speculative backoff. The previous post-restart sample averaged 3267.6 estimated replayed tokens per matched follow-up. The fix retains a completion endpoint in the ordinary GPU cache: share attention/ring history, capture normalized recurrent state with a batched GPU kernel at completion, and release pages into the allocator's evictable LRU pool. There are no periodic decode checkpoint copies or permanently reserved completion slots. Disk backing is created under allocation/admission pressure or suspension; a GPU-resident follow-up does not stage its state through CPU/disk.

## Serving configuration

- Qwen3.8-Flash-Next-NVFP4, unchanged revision fc694b54fb0174e0913e6adf86691ef85a4ead47; TP2/EP, BF16 KV.
- MTP3, existing deterministic concurrent kernels and graph sizes retained.
- 40 GiB GPU KV per rank; 48 GiB logical disk per rank.
- 1920-token physical pages; 7680-token reservation increments.
- 32 maximum sequences, 8192 batched tokens, 262144 context.
- memory_completion_cache=true, native_completion_cache=true, prefix_cache_retention_interval=0, disk_write_policy=pressure.
- Disk checksums, deduplication and restore readback retained.

## Live correctness

The source request had 7264 prompt and 64 generated tokens. Every follow-up reused the expected computed endpoint 7327, leaving 101 of 7428 prompt tokens uncached. C4, C10 and the final serial repeat passed all 15 comparisons against the serial reference, exactly matching generated token IDs and returned logprobs. Global disk read and write counters remained zero. See correctness-1789396071382933339/summary.json and live-correctness.log.

## Live aggregate performance and reuse

A read-only 120.349-second window over the active representative workload recorded 20108 generated tokens: **167.08 aggregate tokens/s**, with 8–10 running requests, zero preemptions, zero disk reads and zero disk writes. This is actual server generation throughput, not prompt throughput or one request's stream rate. It is not a controlled before/after speedup measurement and does not establish recovery to sustained 200+ tokens/s.

The active campaign is 20260914T142720Z-temperature-after-recovery: eight temperature arms (0 twice, then 0.2 through 1.2), medium reasoning, append-only planner/worker histories, finite batch. Configured client ceiling is 40, distinct from the observed running concurrency. Sampling-control reports baseline variant.

In the frozen 80 matched real follow-ups:

| Metric | Current sample |
|---|---:|
| Exact estimated previous endpoint hits | 75 / 80 |
| Mean uncached prompt tokens | 1459.225 |
| Mean estimated new input | 1273.475 |
| Mean estimated replay beyond new input | 185.75 |
| Mean older-prompt replay floor | 79.3375 |

All 80 had semantically matching appended assistant output. These are token-count estimates using matching wire-message prefixes, not raw generated-token identity proofs. Different samples have different new input sizes; compare redundant replay, not uncached input alone. See live-followups.json, live-followup-summary.json and extract_live_trace.py.

Five endpoint misses remain in this sample. Their expected completion snapshots were logged as saved; two fork requests were looked up before their new snapshot's publication. Wire histories also reserialize tool arguments and sometimes tool IDs, so semantic identity alone cannot establish token-prefix identity. Exact attribution between publication timing and rendered-token mismatch has not been established for every miss. No inference-service errors were found in the inspected serving log; startup emitted existing warnings and initial JIT warnings.

## Interpretation and limits

The connector's external-prefix hit counter includes GPU completion hits. It must not be labeled disk transfer: the measured disk-byte deltas are zero. The ordinary KV utilization gauge excludes zero-reference cached pages; low utilization alone does not prove eviction.

Allocator lifetime, lookup pinning, both-rank publication, active suspension planning and isolated GPU-to-disk-to-GPU byte-exact roundtrips passed their component checks. The live test verifies full-model GPU reuse. Full-model forced disk-resume and full-context C20 pressure correctness have not been newly validated. The remaining five endpoint misses and the gap to historical 200+ throughput remain audit items; the measured completion-cache regression is substantially reduced, not a claim that all performance work is finished.
