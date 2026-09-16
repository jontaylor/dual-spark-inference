# Performance selection — provisional, 14 September 2026

Observation deadline: **10:00 London (09:00 UTC)**. Selection discussion: approximately 11:00 London. This report is not final until the deadline and live-service audit. Detailed history: `LEDGER.md`; earlier working report: `SELECTION-before-J-validation.md`.

## Current state

**J is validated for the tested routes and running representative batch-004.** It keeps I3’s MTP5, 1920-token pages and completion retention, independent 64-token completion grid, fixed-reduction kernels, exact-content disk deduplication and inherited full-attention-page sharing. Its only serving-policy change is reducing advance scheduler reservation units from 32768 to **7680 tokens**. This aims to reduce premature spills while retaining concurrent batching. No per-request GEMVs; no request serialization.

Both ranks retain 40 GiB GPU KV, 48 GiB logical disk capacity, 32 sequences and 8192 batched tokens. CPU checksums and post-restore GPU readback remain enabled. All 18 J overrides were verified inside both running containers. Exact rank-specific H and I3 rollback manifests are saved.

The workload controller resumed at 08:10:30 UTC; actual controller state and new batch-004 were independently verified. Its campaign was created at 08:10:30 UTC and observed running by 08:10:40 UTC. No hold remains. Fresh physical observers run on both ranks. J representative throughput and write results are developing.

## Completed representative cycles

| Configuration | Aggregate output tokens/s | Output tokens | Prompt reuse | Requests | Payload written | Payload loaded | Supplementary checks |
|---|---:|---:|---:|---:|---:|---:|---:|
| H: MTP3, 1600-token pages | 125.48 | 422002 | 96.23% | 496 | 84.64 GB | 103.01 GB | 5/8 passed |
| I3: MTP5, 1920-token pages | 126.34 | 517218 | 96.83% | 599 | 148.71 GB | 72.51 GB | 3/8 passed |

Both completed all eight benchmark and original CLI checks, with zero preemptions. Actual supplementary failures retain a stale working-directory cache: relative-path resolution raises `ImportError` after changing directories while the absolute control succeeds. Earlier attribution to egg/zip failures used the wrong artifact; see `QUALITY-CORRECTION.md` and the saved cwd audits. These stochastic workload outcomes do not establish a causal serving regression. Preserve the failures when selecting, rather than treating speed as overall quality.

These are whole sample-defined cycles, including tool waits and declining active-arm count. H lasted 3363 seconds and included a paired readback experiment and probes; I3 lasted 4094 seconds with full verification throughout. Model outputs, context lengths and concurrency differed. **MTP5 has not established a compelling whole-cycle advantage.** Early matched windows were somewhat faster but confounded; the large write increase warrants testing J’s smaller reservation units.

## What the disk evidence establishes

- H: 46.04% of newly observed unique physical payload bytes retired unread; I3: **46.71%**, or 69.47 GB of 148.71 GB. These are confirmed lower bounds before restart cleanup, excluding live unread bytes. Metadata detects any reader, without PID attribution.
- Loaded/stored totals are not a unique-content recall ratio: a blob may be loaded repeatedly. H loaded more than it wrote despite substantial unread retirements.
- Exact-content deduplication already avoids duplicate physical payload: approximately 65–66% in mature H observations and 51.22% across the completed I3 cycle (55.61% at its 35-minute snapshot). It preserves logical checkpoints. Inherited complete-page sharing also reduces logical duplication, copies and hashes where provenance is known.
- I3 worker writes were about 50.54 GB per rank over a measured 39-minute interval; devices wrote about 51–52 GB. Cancelled worker writes were negligible. Device counters include other activity, but filesystem buffering is not demonstrated to eliminate most writes.
- Some useful blobs were read within seconds of writing. A blanket delay would withhold useful cache state unless RAM retained it. J instead tests less aggressive advance reservation, so fewer resident checkpoints may need spilling.
- Fragment cleanup remains **undeployed**. CPU ownership/allocator/lookup checks passed, but its missing-checkpoint trigger had zero opportunities in the measured I3 snapshot. No demonstrated live saving justifies mixing it into J.

## Determinism and correctness scope

J passed normal and fixed-seed serial/concurrent repeats (10 requests each), 12 mixed comparisons with observed peak eight running, and warm C1/C4 cache comparisons: exact tokens and returned full scores. Generation across reservation boundaries 7680 and 15360 also passed cached C1/C4 equality with no preemptions. The first growth harness failed on the wrong JSON field; preserved responses matched, and the corrected harness additionally checks returned prompt IDs against the supplied token list.

**Cold versus cached execution still differs in both tokens and scores.** Archived H and J have exactly equal prompts, tokens and scores when each route is compared separately, including the same within-configuration cold/warm divergence. I3’s short probe happened to retain equal tokens but different scores. None provides universal cache-independent or cross-restart determinism. J disk-versus-resident equality passed: 325877760 bytes read versus zero on an independent resident reconstruction, with exact prefix, tokens and scores. Initial chosen targets remained resident; those coverage failures are preserved separately.

The original HC/cuBLAS concurrency effect reproduced in the pristine image and was not caused by our patches. An earlier completion-cache boundary eligibility/save mismatch was caused by our changes and repaired with the latest accepted 64-token-grid checkpoint.

## Verification-cost decision

Completed eight-window ABBAABBA comparison: full GPU readback **164.50 aggregate tokens/s**, disabled **161.72**. All cached probes passed; neither preempted. Workload trajectory differed across windows, so this is descriptive, but it demonstrates no throughput reason to remove the extra check. **Keep full verification enabled.**

## Evidence

`batch-002-counters.json`, `batch-003-counters.json`, `H-final-workload-quality.json`, `I3-final-workload-quality.json`, `I3-final-unread-write-lower-bound.json`, `I3-workload-storage-io.json`, `H-J-archived-cache-route-comparison.json`, `validation-J/summary.json`, `growth-validation-J/summary.json`, `J-runtime-verified.json`. Live counters: `metrics.jsonl`; workload state: `cycles.jsonl`; physical device/worker I/O: `storage-io.jsonl`.

## J matched observation at 32½ minutes

Matched elapsed H / I3 / J: confirmed retired-unread payload **25.75 / 41.88 / 17.11 GB**, respectively **40.47% / 47.30% / 31.63%** of written payload. New blobs must both attach and retire inside each metric window; live unread bytes and preexisting data are excluded. This is a conservative measured lower bound, not an oracle admission-policy saving. Workload trajectories and concurrency differ.

J was delivering 144.77 aggregate tokens/s, 95.18% prompt reuse, 54.10 GB written and 23.30 GB loaded, with zero preemptions; all eight quality outcomes remained pending. The separate fragment-cleanup candidate had only three non-missing page references across eighteen broken-checkpoint matches, with sharing/ownership unresolved, so no live deployment was justified.

Evidence: `H-I3-J-unread-matched-elapsed.json`, `H-I3-J-matched-elapsed.json`, `J-checkpoint-fragments-latest.json`. Time-series export: `workload-comparison.png` (regenerated at final audit).

## Supplementary-quality correction

The actual supplementary probe is the relative-path/cwd audit, not `verification.txt`. H5/8 and I3 3/8 pass counts remain correct; earlier explanations using egg/zip failures were wrong. J currently has1pass/2fail/5pending. These failures are incomplete generated repairs, and must be retained in the selection. See `QUALITY-CORRECTION.md`.
