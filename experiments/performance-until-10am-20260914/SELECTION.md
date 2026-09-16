# Performance selection — observation complete

The requested monitoring window ended at **10:00 London / 09:00 UTC, 14 September 2026**. **J remains serving; the frozen workload continues with no controller hold.** All measurement processes ended normally. Final runtime and health audits passed. Selection is expected around 11:00 London.

**J has not demonstrated a decisive advantage over H.** It substantially reduced observed writes relative to I3, with lower throughput. Against H, throughput and written payload were similar. Keep these descriptive comparisons separate from causal claims; the workload trajectories and concurrency differed.

## Matched elapsed observations

Approximately 49minutes from each workload start, including tool/client waits and declining active-arm counts. H/I3 windows 2952.7seconds; J 2962.3seconds. All had zero preemptions. Units below are decimal GB of filesystem payload, not NVMe device I/O.

| Configuration | Aggregate tokens/s | Prompt reuse | Written GB | Loaded GB | Confirmed retired unread GB (% of written) |
|---|---:|---:|---:|---:|---:|
| H | 133.15 | 95.92% | 80.52 | 94.78 | 37.84 (46.99%) |
| I3 | 143.52 | 96.50% | 127.42 | 47.09 | 59.96 (47.06%) |
| J | 131.83 | 96.21% | 82.88 | 46.71 | 23.84 (28.77%) |

H uses MTP3 and 1600-token pages. I3 uses MTP5 and 1920-token pages. J keeps I3 and reduces advance reservation units from 32768 to 7680 tokens. Mean sampled running requests were H 8.24 / I3 9.40 / J 7.01. H includes the paired readback experiment and its probes. J's campaign was created 0.34 seconds after the pre-release baseline. These differences prevent attributing all observed changes to the configuration.

J produced 390527 output tokens and completed 424 requests during the window. It is still an incomplete cycle: three arms remain running. H's completed cycle averaged 125.48 tokens/s; I3's completed cycle 126.34 tokens/s. Do not compare those whole-cycle figures directly with J's partial window.

## What changed and what remains running

J retains concurrent batching, scoped fixed-reduction BF16 GEMMs, deterministic GDN routes, MTP5, an independent 64-token completion checkpoint grid, exact-content disk deduplication and inherited complete-attention-page sharing. **No per-request GEMVs or request serialization.** Each rank has 40 GiB GPU KV and 48 GiB logical disk capacity; 32 sequences, 8192 batched tokens.

Smaller reservation units avoid setting aside excessive future attention capacity, which can otherwise force early checkpoint spills. Actual scheduler geometry and growth through 7680/15360 tokens passed validation. All 18 overrides were independently verified inside both running containers, with unchanged container starts. Full CPU checksums and GPU readback remain enabled; recent actual read events on both ranks confirmed readback verification, with no logged errors.

Exact rank-specific `rollback-H-r0/r1.json` and `rollback-I3-r0/r1.json` are preserved. The separate fragment-cleanup candidate remains undeployed: its observed opportunities were too small/uncertain to justify another restart and validation cycle.

## Selective writes and recalls

J's exact-content deduplication avoided **53.59%** of requested logical storage payload over this workload window. It preserves logical entries rather than discarding potentially useful checkpoints. Inherited complete-page sharing additionally avoids some duplicate logical entries, copies and hashes.

The matched retirement result is a conservative lower bound: only new blobs both attached and retired inside the metric window are counted. Live unread blobs and preexisting blobs are excluded; metadata identifies any reader, without PID attribution. J still wrote at least 23.84 GB that retired unread. This is observed unused content, not a guaranteed saving available to an online admission policy.

Loaded/stored totals are not a unique-content recall percentage: one blob can be read repeatedly. Also, the dashboard's external prefix cache includes GPU-resident completion snapshots and disk. Local/external query percentages use different denominators; use prompt-usage counters for combined reuse.

Worker/device observations show substantial actual storage writes and negligible cancelled writes. Buffering did not eliminate most writes. Some useful content is read within seconds, so blanket delayed availability would lose useful hits unless RAM retained it.

## Correctness and task quality

J passed tested warm-cache C1/C4 token and returned-score equality; normal and fixed-seed repeats; 12 mixed comparisons with peak 8 running; reservation-growth probes; and a real disk-versus-independent-resident oracle (325877760 bytes read versus 0, exact prefix/tokens/scores). Initial disk targets stayed resident; those coverage failures and the corrected growth-harness JSON-field error are preserved.

**Cold versus cached execution still differs in tokens and scores. Universal cache-independent or cross-restart determinism is not established.** Archived H and J matched exactly when each cache route was compared separately, including the same within-configuration cold/warm divergence.

At the final audit, J had five completed arms: all five passed benchmark and original CLI checks; supplementary verification had **1 pass / 4 fail**, with three arms pending. The failures leave a stale working-directory cache and raise ImportError on relative-path resolution after changing directories while the absolute control succeeds. H completed 5/8 supplementary passes; I3 completed 3/8. These small differing trajectories do not establish a causal serving regression, but the incomplete repairs remain material quality failures.

**Correction:** earlier explanations attributed supplementary failures to egg/zip output from the wrong artifact. The actual supplementary source is `operator-audit/relative-path-cwd.json`; all final J pass/fail flags were checked against its exit status and parsed result. See `QUALITY-CORRECTION.md` and `*-cwd-supplementary.json`. Historical ledger entries are superseded by this correction.

The pristine-image attribution reproduced the original HC/cuBLAS concurrency issue, so that initial issue was not caused by our patches. An earlier completion-cache boundary eligibility/save mismatch was introduced by our changes and repaired with aligned accepted-state checkpoints.

## Selection evidence

`FINAL-AUDIT.json`, `H-I3-J-final-matched-elapsed.json`, `H-I3-J-final-unread-matched-elapsed.json`, `J-final-disk-summary.json`, `J-final-workload-quality.json`, `J-actual-supplementary-final.json`, `J-workload-storage-io.json`, `J-runtime-verified.json`, `J-final-health-audit.json` and `LEDGER.md`.

The exported curves are `workload-comparison.png` and `.svg`. The paired readback ABBAABBA experiment found no demonstrated aggregate benefit from disabling GPU readback (164.50 tokens/s full versus 161.72 disabled); full verification was retained.

## Subsequent historical-baseline finding

The dashboard review recovered the earlier September13 temperature-sweep counters. Its first approximately49minutes averaged178.08tokens/s with9.42mean running requests, compared with H133.15/8.24,I3 143.52/9.40,J131.83/7.01. Frozen source hashes and workload conditions match. Thus H/I3/J alone omit the earlier faster baseline; they do not establish restoration of original throughput. Computation-route cost remains unisolated, and generated trajectories still differ. Detailed evidence: `../dashboard-history-20260914/REPORT.md`.

## MTP selection correction

Subsequent audit: keeping MTP5 did not establish a depth optimum. H/I3 completed-cycle throughput differed by only about 0.7%, with other configuration/workload differences; no controlled MTP4 comparison was found in this campaign. The user's prior MTP3/MTP4 boundary results were not overturned. MTP5 remains running but is an unproven performance candidate. See ../dashboard-history-20260914/MTP-SELECTION-AUDIT.md for acceptance/cost thresholds and the slower-decode amortization hypothesis.

## Superseded by user-requested MTP3/event-checkpoint deployment

J/MTP5 is no longer the live configuration. The user requested returning to MTP3 and removing expensive periodic decode snapshots. The completed candidate saves at completion/suspension events and supports exact unaligned restore boundaries with a fixed partial first chunk. See ../completion-triggered-20260914/RESULT.md and FINAL-AUDIT.json for authoritative configuration, correctness, performance and coverage limits. Do not describe J as current based on the earlier historical text above.
