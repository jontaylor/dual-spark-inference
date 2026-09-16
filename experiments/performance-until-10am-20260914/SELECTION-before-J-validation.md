# Performance selection — working evidence, 14 September 2026

Monitoring/optimisation ends at 10:00 London (09:00 UTC). Selection discussion is around 11:00 London. This report is provisional until the monitoring deadline and final runtime audit.

## Current validated option

**H: MTP3 + scoped fixed-reduction kernels (fine decode tiles through M=128) + aligned completion checkpoints + exact-content disk deduplication + inherited full-attention-page sharing.** 40 GiB GPU KV and 48 GiB logical disk budget per rank; max 32 sequences, 8192 batched tokens. CPU transfer checksums remain enabled. Full GPU readback is the default; the completed ABBAABBA comparison found no demonstrated aggregate benefit from disabling additional post-restore readback, and full readback is restored on both ranks. Concurrent execution; no per-request GEMV.

C is the rollback. H (inherited-page sharing plus measured fine GEMMs, full verification) finished its measured cycle and is the latest validated rollback. I failed its QSA ring/block divisibility check. I2 passed layout setup but failed our scheduler’s explicit three-draft guard. I3 (I2 plus the audited five-draft guard) is loading and not yet validated. Seven baseline arms finished; the interrupted final arm hit a 660-second tool timeout after resumption and is excluded from clean comparisons. A fresh eight-arm cycle is running on H. The earlier controller hold was released at 05:06:06 UTC. A new controller-only hold began at 05:24 UTC for the next MTP5 transition: active arms and tools continue normally; only admission of the next cycle is held. `mtp5-boundary-status.json` records the watchdog and automatic release deadline.

## Measured results and limits

- The first 37.9 minutes of batch-001 produced 300,687 output tokens at **132.34 aggregate tokens/s**, with **95.53% prompt-token reuse**, 291 completed requests and zero preemptions. This is an in-progress interval, not an A/B speedup measurement. Later cycle phases have fewer active arms and must be reported separately.
- The corresponding disk-event snapshot requested 165.8 GiB of logical storage but wrote 56.3 GiB of new payload: **66.06% avoided** by exact-content deduplication. 30.1 GiB was loaded. These are filesystem payload bytes, not NVMe device I/O.
- Unique-blob lifetime observation confirms nonzero writes deleted without any read. Use `physical-recall-summary.json` for timestamped counts. Do not call all live unread bytes waste; early retirements overrepresent short-lived blobs. Preexisting blobs and new generations are separate.
- Three large zero-cache requests retained **44,231 / 47,972 / 46,154 identical prefix tokens**, respectively 97.43% / 96.65% / 97.16% of their new prompts. Actual chat-render token counts match original inference usage. The generic `/tokenize` test did not match and is excluded.
- Those branch points had prior saved checkpoints at 44,224 / 47,936 / 46,144 tokens. Losing reusable ancestor state causes expensive prefill. Exact cause of each loss (record index versus one or more component pages) is not yet instrumented in C.
- MTP3 accepted 1.927 draft tokens per draft step in a measured batch-001 interval. Acceptance at draft positions 1/2/3 was 79.88% / 62.86% / 50.00%. MTP5 later-position acceptance and end-to-end benefit are unmeasured.

## Candidate options

| Option | Intended benefit | Evidence | Trade-off / outstanding gate |
|---|---|---|---|
| C, rollback | Avoid duplicate disk payload while retaining all logical checkpoints | Full/seed C1/C4, mixed load and independent disk-vs-resident oracle passed | Logical entries still consume separate capacity; full copy/hash/readback costs retained |
| Inherited-page sharing on C | Share storage keys of complete attention pages actually restored; reduce logical entries, copies and hashes | CPU provenance/exclusion/fallback/pin tests passed; actual allocator fixture retains 5 vs 2 checkpoints in 16 slots (synthetic); both-rank sources staged/hashed | H full-model/cache/disk checks passed; three inherited full pages exercised with exact continuation. Locally supplied GPU pages remain excluded |
| Fine fixed-K GEMM | Faster HC-down/GDN-ba decode while preserving current prefill plan | Refined offline sweep: all 96 cases exact, including all-row equality against baseline tiles | Offline 96-case all-row equality and H full-model checks passed; H real-workload speedup still unmeasured |
| Optional GPU readback | Remove extra post-restore GPU-to-CPU copy and second hash | Real-file/simulated-DMA checksum/corruption/control tests passed | Always retains checksum before GPU restore. Disabling readback removes detection of a transfer fault after that check; Completed eight-window comparison: 164.50 tps full vs 161.72 disabled; full verification retained |
| I: H + MTP5 | More output per verification step | 84 CPU checkpoint-selection cases passed at depth 5; H-based manifests prepared with only depth and graph-size changes | Not deployed. GPU recurrence, model equality, acceptance and memory/throughput tests pending; waiting for clean cycle boundary |
| Packed state spans | Avoid padded disk bytes | Live full-blob geometry checked on every group/rank | Rejected for this layout: no smaller payload demonstrated; no reload justified |

## Correctness and workload quality

Tested concurrency equality is scoped to the measured routes. Cold versus cached replay, restored versus uninterrupted execution, and equality across restarts/configurations are not established; some differ. Do not claim unconditional determinism.

Original pristine-image attribution reproduced the HC/cuBLAS concurrency effect, so that initial numerical problem was not introduced by our patches. The earlier completion-cache regression was caused by our boundary eligibility/save mismatch and was repaired by retaining the latest accepted 64-token-grid state.

The repeated workload includes tool execution and varied sampling. Preserve each arm's benchmark, original CLI and supplementary result. Some batch-001 arms pass benchmark/original CLI but fail supplementary checks; speed and numerical probe success do not establish all-task quality success. Cycle 0 spans changes/pauses and cannot serve as a clean single-configuration timing baseline.

## Evidence

`LEDGER.md`: chronological decisions, deployments, failures and process identities. `metrics.jsonl` and `cycles.jsonl`: continuous counters and workload results. `cold-prefix-render-check.json` and `cold-ancestor-saves.log`: branch-return evidence. `physical-recall-summary.json`: unique physical generations. Candidate-specific `validation-*/summary.json`: live gates when executed. Retain failed coverage checks alongside successful replacement oracles.

## H observation at 05:21 UTC

The first full-readback paired window delivered 139.29 aggregate tokens/s over 120 seconds, with 543,129,600 bytes loaded and 3,801,907,200 bytes stored. One window does not establish a readback cost or H speedup. Phase 1 is measuring with readback disabled; its cached token/score gate passed.

At the latest physical-observer snapshot, each rank had 436 new live unique blobs (9,866,854,400 bytes), of which 41 (927,846,400 bytes) had an observed read. No new H blobs had retired, so there is no H unread-at-retirement rate yet. Preexisting validation blobs are excluded from that conclusion. Live unread bytes may be recalled later.

At 05:25 UTC, H batch-002 had 168,214 generated tokens over 1181 seconds (142.43 aggregate tokens/s), 94.18% prompt reuse, 190 completed requests, zero preemptions, 38,562,201,600 payload bytes stored and 4,752,384,000 loaded. This is an unfinished cycle spanning both verification modes. The new H physical retirement cohort reached 50 blobs per rank, all unread (2,263,040,000 bytes combined), with lifetimes roughly 319–439 seconds. This early-retirement cohort does not establish the unread rate of all H writes.

## Completed readback comparison, 05:32 UTC

Eight ABBAABBA windows of 120 seconds completed; all eight cached token/full-score probes passed and no preemptions occurred. Full GPU readback: 164.50 aggregate tokens/s; readback disabled: 161.72. Summed transfer seconds/GiB were 7.27 versus 4.80. Context, concurrency, cache traffic and draft acceptance varied, so these descriptive differences are not isolated causal estimates. There is no demonstrated aggregate throughput benefit from disabling readback. **Decision: keep full verification enabled.** Both live container control files independently verified true after completion (`readback-paired-H/restoration-verified.json`). CPU checksum was always retained.

## Physical I/O follow-up

The larger H physical cohort confirmed 24,169,267,200 bytes retired unread out of 61,962,035,200 new unique bytes observed (39.01% lower bound at 05:38 UTC). Retired read blobs had recorded access within 0.52–8.10 seconds of writing. This rules against simply deferring availability; retaining a bounded RAM copy could still serve immediate reads. Buffered filesystem I/O already uses the page cache.

The new 61-second worker/device observation recorded 1,199,411,200 worker write bytes per rank and approximately 1.21 GB of device writes per rank, with zero cancelled worker writes. Device totals include unrelated host activity; this is not retrospective attribution of all stored payload to SSD writes. `storage-io.jsonl` tracks further intervals and process identities.

## Matched elapsed baseline comparison

First 37.9 minutes: C 132.34 aggregate tokens/s, H 141.01; cache reuse 95.525% versus 95.490%. Mean sampled running requests were 7.785 versus 9.031, and uncached prompt tokens 431,718 versus 510,555. Both had zero preemptions. H also includes the paired readback experiment. This is descriptive aggregate performance under different trajectories/concurrency, not an isolated 6.6% speedup claim. See `C-H-38minute-comparison.json`.

## Additional candidate: broken-checkpoint cleanup (not deployed)

Six matched unusable checkpoints retained 37 page references after partial eviction. Candidate `completion.fragment_cleanup.py` removes unreferenced idle remnants when lookup detects a broken checkpoint, protecting valid, selected, ready and pending checkpoints plus active inherited-page provenance. Three helper ownership/fallback checks pass; full integration and live gates remain outstanding. It is separate from I/MTP5. The references may be shared, so 37 pages is not a claimed reclaimable amount.

## H completed cycle and I startup

H batch-002 finished at 06:02:02 UTC: 422,002 generated tokens over 3363.07 seconds, 125.48 aggregate tokens/s, 96.229% prompt reuse, 496 completed requests, zero preemptions, 84.64 GB payload written and 103.01 GB loaded. Includes the paired verification-mode experiment. Per-arm quality is in `H-final-workload-quality.json`.

I deployment began at a clean boundary with no arm owners or backend work. First startup failed before model loading because free disk was ~25 MB short of the configured reserve; clearing disposable pip downloads provided ~937 MB and the unchanged candidate started at 06:05:43 UTC. All 17 overrides verified inside both containers. Primary validator session98071 and disk-chain31666 are pending. Exact H rollback manifests are saved. Controller-only hold remains active until validation/release; no workload arm was paused.

## Revised MTP5 layout (I2), 06:18 UTC

I failed after weight loading: MTP5 increased the maximum recurrent-state page to 1,654,784 bytes, causing the 1600-token block setting to round up to 3200, which is incompatible with the 12-slot QSA ring. The failed startup was stopped. A CPU fixture using actual model shape calculators and cache specs reproduces 3200 and 98.02% padding; block 1920 yields 18.81% padding and divides the ring/backend/checkpoint alignments.

I2 changes both block_size and prefix_cache_retention_interval to 1920, preserving the independent 64-token completion checkpoint grid and all 17 source overrides. Startup began 06:17:20 UTC. Primary39377 and disk-chain30143 remain pending. This layout change is part of the MTP5 trade-off, so a future performance result cannot be attributed only to extra draft tokens. Exact H rollback remains saved.

## Scheduler depth gate and I3, 06:29 UTC

I2 reached scheduler initialization and failed the existing targeted-invariance guard limiting MTP to three drafts. Startup was stopped immediately. The 17 overridden files were audited for explicit depth-three assumptions. The fixed recurrent GPU kernel then passed six-token C1/C4 output/state equality and equality against six separate recurrence steps for all accepted-state indices 1–6. CPU checkpoint selection and actual scheduler method/depth/alignment guard checks also passed.

I3 extends that scheduler guard to five drafts, preserving the MTP-only and 64-grid restrictions; it retains I2’s 1920-token layout. All 17 overrides verified inside both new containers. Primary32542 and disk-chain42355 are running. These component checks do not establish full-model correctness or performance. No fragment-cleanup code is included.

## I3 validated and workload resumed, 06:41 UTC

Primary gates passed: normal and fixed-seed C1/C4 repeats (10 each, 117 tokens), 12 mixed comparisons with peak eight running, and the 7360-token warm-cache boundary all had exact tokens and scores. Cold versus warm scores still differ. The long-target disk test read 380,190,720 bytes and matched; an independent disk-versus-resident oracle read 325,877,760 versus zero bytes and matched prefix, tokens and scores. These establish the tested routes, not universal determinism.

Controller hold released and actual controller state S/new batch-003 verified. MTP5 is live; comparative performance remains pending. New physical observers use physical-I3 and preserve the H epoch. Full verification remains enabled.

Final H write-lifetime evidence before restart cleanup: 38,969,548,800 of 84,637,696,000 newly observed physical bytes retired unread, a 46.04% lower bound. No observer errors. Live unread bytes excluded; this is not a demonstrated admission-policy saving. Whole-cycle loaded payload exceeded written payload despite these unused writes, because byte totals do not identify unique reused content.

Checkpoint-cleanup CPU integration now passes the actual candidate lookup with DiskSlotManager: a broken longer checkpoint falls back to a valid shorter one, orphan pages are reclaimed, shared/inherited pages stay available, and pending stores become usable on completion. See fragment-lookup-checks.log. Candidate remains undeployed; live restore and performance remain untested.

## Early I3 storage trade-off, first 10.3 minutes

H: 127.06 aggregate tokens/s, 4.25 GB payload written, 362 MB loaded, mean 7.71 running requests. I3: 131.40 tokens/s, 20.31 GB written, 380 MB loaded, mean 9.94 running. Completed requests were 92 versus 132. The much earlier I3 spilling warrants watching; page size increased and concurrent workload trajectories differ. This is an early interval, not a causal or final cost comparison. H-I3-first10minute-comparison.json preserves the evidence.

## Reservation-step candidate (not deployed)

The running scheduler reserves attention growth in 32,768-token units and spills against ledger free space. Prepared parking_scheduler.reservation.py makes this configurable, retaining the current default. A 7680-token step is four I3 attention pages. Actual ParkingPolicy CPU checks cover every context length through 262144 for both layouts, eight concurrent admissions, growth, two-rank save/restore acknowledgements and cache pressure. Full scheduler/model validation and real write savings remain unproven.

## I3 at 35.4 minutes (provisional)

Matched elapsed H/I3: 143.57/149.05 aggregate tokens/s; 95.32/95.71% prompt reuse; mean running requests 9.22/10.41; payload written 67.44/93.64 GB; loaded 14.48/28.62 GB. Neither preempted. H includes readback experiment/probes; trajectories differ. I3 deduplication has risen to 55.61% of logical payload, so its first-spill 27–28% was not representative of the developing cycle. Quality and full-cycle performance remain pending. See H-I3-first35minute-comparison.json and I3-workload-latest-disk-summary.json.

I3 physical-I/O observation over 39.36 minutes (starting 17.56 s after workload launch): each unchanged worker wrote about 50.54 GB; host devices wrote 51.27 and 52.19 GB. Cancelled writes were only 356,352 bytes per worker. Device totals include other activity, but buffering did not appear to eliminate most writes. See I3-workload-storage-io.json.

First I3 completed arm (temperature 1.0): benchmark and original CLI passed; supplementary verification had 2 failures and 45 passes. Both failures call removed zipimporter.find_module in egg/zip tests, the same failure class observed in H temperature 1.2. Preserve this quality failure; it does not establish a causal inference regression. Exact output: I3-temperature1p0-supplementary.txt. Seven arms were still running at this snapshot.

## I3 completed cycle; J deployment issued

Batch-003 finished 07:49:06 UTC: 517,218 generated tokens over 4093.77 seconds, 126.34 aggregate tokens/s, 96.83% prompt reuse, 599 requests and zero preemptions. Payload written 148.71 GB, loaded 72.51 GB. All eight benchmark/CLI checks passed; three supplementary checks passed. Five failures were the same removed zipimporter.find_module egg/zip cases, including one temperature-zero arm. H completed at 125.48 aggregate tokens/s with 84.64 GB written, 103.01 GB loaded and 5/8 supplementary passes. These are different workload trajectories; I3 has not established a compelling whole-cycle advantage.

J deployment began only after all owners exited and backend idle was verified. Exact I3 rollback manifests and final physical observers were preserved before service cleanup. J keeps MTP5 and changes only reservation granularity through one added scheduler override and launcher option. Model/growth/disk validation remains pending.
