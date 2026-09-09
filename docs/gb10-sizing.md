# GB10 sizing and placement experiment — 2026-09-09

Status: both experimental variants passed their correctness checks, but
neither established a dependable overall serving improvement. Retain the
previous validated deployment, `gb10-v0.29.0-2026-09-09`. The experiments and
all numerical results remain available on this branch; no new tuning from
this experiment is selected for the live service.

This experiment keeps the official Qwen FP8 checkpoint, BF16 KV, TP2+EP+MTP3,
C4, 262144 total tokens/request and the 8192-token scheduler budget. It tests
hardware sizing choices on the validated vLLM 0.29 bundle. The prior release
and images remain available for rollback.

## Candidate changes

| Area | Implementation | Activation |
|---|---|---|
| Aligned prefill remainder | Scheduler code allocates whole 1600-token quanta and rotates the extra quantum. Four long prefills can schedule 8000 tokens instead of 6400; decode and speculative input positions are reserved. Existing admission, allocation and checkpoint rules still apply. | Experimental. `optimizations.prefill_remainder` / `GB10_PREFILL_REMAINDER=1`, with fair prefill and Mamba alignment |
| BF16 matrices | 18 additional measured plans for existing kernels, including missing M3/M12 shapes and two M16 shapes. Five marginal/noisy cases retain the standard fallback. | Existing `optimizations.bf16_kernels`; 58 total plans |
| QSA scoring | Static launch profile per captured row count, chosen across 12K/65K/262K. | `optimizations.qsa_score_profile` / `GB10_QSA_SCORE=1` |
| Sparse attention | GB10 block/split/warp profiles for TP2 rows 4/8/12/16, with the existing FP32 split merge and attention arithmetic. | `optimizations.qsa_sparse_profile` / `GB10_QSA_SPARSE=1` |
| Cache diagnostics | Report actual bound tensor shapes, dtype, byte strides and layer counts once at startup. | Existing `GB10_KV_ACCOUNTING=1` |

QSA profiles are limited to measured BF16 TP2 shapes on SM121. Other shapes
use the existing policy. The profile cannot depend on a CPU read of context
length inside a captured decode graph, so one profile is checked across all
three context lengths. Row 4 is also checked with four separate draft requests.

## Measurement controls

The before/after serving suite uses frozen prompts and leading identities,
two engineering tasks, two repetitions/task and 1024 output tokens/request at
C1/C2/C3/C4. C4 then runs cold and three warm repetitions at 65K and 262K total
context. A mixed workload injects three cold 65K prefills into one ongoing
decode. It checks output markers, prefix hits, preemptions, MTP acceptance,
GPU clocks/power/temperature and both nodes' available memory. Private prompt
and response files stay outside public Git.

The API does not expose a working reset-prefix-cache route. Long-context
cases therefore use unique fixed leading identities, and the short cases are
explicitly warmed. Each serving build starts with a new cache. The corpus is
fixed; its SHA-256 is
`eb3d84935ca1a7fcd1ab3ccdf7f44c630c9e2c0945a43c158799817d7775e0ec`.

Isolated matrix tests screen 64 MiB of distinct weights, then validate fresh
128 MiB weights with five alternating paired trials. Their FP32 errors are
checked relative to BF16 linear. QSA tests touch at least 64 MiB of distinct
cache data and use the 0.29 BLNHC layout, random physical pages and full graph
page-table capacity. Confirmation uses a new seed, seven alternating pairs,
full top-k selection comparison and an independent FP32 sparse reference.
The extended upstream QSA reference suite passed all 15 selected cases.

The proposed sparse profiles reduced isolated target-call time by about
8–35%; matrix additions reduced their individual call times by about 6–23%.
Scoring helps shorter contexts, while full-context C2–C4 selection is roughly
neutral. These percentages cannot be added or treated as serving speedups.
The source repository records full numerical kernel results in
`gb10/provenance/sizing-kernels-20260909.json`.

## CPU placement

Each L3 cluster contains five A725 and five X925 cores. Fast cores are Linux
CPUs 5–9 (8 MiB cluster L3) and 15–19 (16 MiB cluster L3). Both nodes' sysfs
confirms 512 KiB private L2 on A725 and 2 MiB on X925. Current driver limits
are 2.808 and 3.9 GHz respectively. Extra memory/SLC connectivity for cluster 1
has not been established.

The gather implementation parallelises missing-row reads. Cache-hit copies
and cache insertion are serial. The worker-node sweep compares whole clusters,
fast-core groups and unrestricted placement, using byte-exact real n-gram IDs.
Cold-page cases evict only the selected table pages; no global cache dropping
is used. Software-cache misses on resident file pages and actual NVMe faults
are measured separately. Whole-process/OpenMP affinity is an experimental
control and should not be applied blindly to the serving engine's threads.

All 52 placement/thread profiles completed with byte-exact results, using
reversed order for the repeat. Retain eight gather threads and unrestricted
placement. On matched eight-thread repeats, the fast-core-only group took
19.59 ms versus 6.57 ms unrestricted for a 25,600-row resident-file replay,
and 10.04 ms versus 5.96 ms for 256 selected cold-page misses. The largest
128,000-row replay favoured the fast-core group (13.79 versus 16.15 ms), so
there is no universal placement winner.

On the unrestricted profiles, a single thread was faster for small resident-file
software-cache misses, but 256-row cold-page misses took about 20.3 ms with
one thread versus 5.9 ms with eight. Fully warm software-cache calls were only
a few microseconds at 64–256 rows. A row-count-only rule would trade away
parallel NVMe fault handling, so no thread-count/affinity change is selected.
The numerical aggregates are in `gb10-sizing-cpu.json`.

## Combined candidate runtime identity and rollback

The measured runtime code is source commit `dc823ba936`; `0f3007d27e` only
adds the live-layout confirmation to its numerical provenance. Both running
containers matched the expected SHA-256 values for all 37 tracked runtime
Python modules and the complete 58-entry BF16 plan file. The latter hash is
`54b5e70c95da12888bbfac30767e31baa43db871a55e1930dac8b49b4ddcf6f4`.

For this test, each node's previously validated image was updated incrementally
with the changed Python modules and plan data, preserving its native libraries.
The head image is
`sha256:2e6aca6250c312039fda071b13b3220f54df0353c80eb16bf3c9a1f721233965`;
the worker image is
`sha256:16c9edbf2127637038a7fe1b55c9c99d7533aa7986fae9516a5cf52c0bcadea6`.
Their distinct image IDs are recorded rather than presented as one identical
image. The public Dockerfile builds the pinned source from the official 0.29
ARM64 base; a fresh build needs its own image and serving checks.

Startup on both ranks confirmed a 22,630,400-byte physical block stride,
1,024-byte main-cache token stride and 256-byte compressed-cache token stride,
matching the isolated benchmarks. There are 1,025 usable physical blocks and
1,423,067 reported logical KV tokens. The raw recurrent-cache allocation is
reported as byte storage (`int8`); this is not a change to recurrent-state
precision. Model loading reported 65.75 GiB per rank. The available KV budget
was 23.58 GiB on the head and 21.64 GiB on the worker; those are profiling
budgets, not equal per-rank allocation measurements. The live physical pool's
1,026 blocks including the null block, multiplied by its byte stride, imply
21.624 GiB. Graph profiling reported 0.45/0.41 GiB for head/worker, followed
by live capture deltas of 0.21/0.23 GiB. These capture deltas are not a census
of every workspace or other allocation retained by the process.

The host-local `95-sizing-candidate.conf` service overrides used for these
experiments have been renamed with a `.disabled` suffix on both nodes. The
untouched `90-v029-candidate.conf` selections restore the previous checkout
and images. The retained release is `gb10-v0.29.0-2026-09-09`, with head image
`sha256:e49126e6768d0ef0edb39208080569af9408565d11afd55595aa7b33ff359ad1`
and worker image
`sha256:230fdcc1d396ce64f9167c9208476b96baaa9f6832a837d4595b0a3a9cbff67d`.
The model, port, credential file and aliases are unchanged.
The restored deployment passed readiness and API smoke checks on both original
image identities. Its fresh startup reports 1,421,680 logical KV tokens, with
the existing C4 and 262,144-token-per-request limits. No benchmark workloads
remain queued by this experiment.

## Serving comparison

This full before/after suite measured the combined candidate, with the
prefill-remainder option enabled. A second experimental variant disabled only
that option; its targeted results are recorded separately below. Neither
variant is selected for the live service.

The baseline is the validated 0.29 PR bundle, not the pre-migration build.
See `gb10-sizing-baseline.json` for its numerical runs and aggregate ranges,
and `gb10-sizing-combined.json` for the complete safe numerical comparison,
sensor summaries, runtime hashes and output-check summaries.
Both serving suites passed the checked output lengths and long-context markers
without preemptions.

| Warm workload | Baseline decode tok/s | Candidate decode tok/s | Change |
|---|---:|---:|---:|
| C1, 12K | 46.66 | 47.17 | +1.1% |
| C2, 12K | 76.28 | 76.10 | −0.2% |
| C3, 12K | 98.96 | 98.48 | −0.5% |
| C4, 12K | 113.14 | 113.90 | +0.7% |
| C4, 65K | 118.16 | 117.07 | −0.9% |
| C4, 262K | 108.35 | 110.77 | +2.2% |

There are four short runs per concurrency and three warm long-context runs
per build. Warm full-context ranges are 107.10–109.82 before and 108.00–112.75
after; warm 65K ranges are 116.96–119.22 before and 116.06–118.06 after.
All these ranges overlap. Short and 65K decode are effectively unchanged in
this comparison; the full-context result is a modest observed improvement,
requiring independent paired repeats to establish a dependable advantage.

| Cold C4 workload | Baseline median TTFT | Candidate median TTFT | Change |
|---|---:|---:|---:|
| 65K | 101.76 s | 98.27 s | −3.4% |
| 262K | 475.32 s | 473.30 s | −0.4% |

Each cold entry is one four-request batch. Full-context prefill is effectively
unchanged. Cold full-context decode measured 107.59 before and 110.55 tok/s
after (+2.7%), also a single batch per build.

Warm full-context MTP acceptance was 62.80% before and 64.26% after. Its
contribution to generated throughput is part of the result; the observed gain
cannot be attributed entirely to faster kernels. Cold acceptance was 64.59%
before and 64.08% after. The original end-to-end repair workload has not been
rerun, and this comparison does not establish recovery of the entire earlier
migration regression.

These are repeated measurements on one startup per build, not independent
machine/restart trials. Output trajectories and temperatures can differ.
The short candidate runs were warmer; average decode clocks differed by less
than 0.5%, and results are not clock-normalized. The suite is a bounded
correctness and performance check, not a repair-quality evaluation or a
measurement of achieved DRAM bandwidth.

## Mixed workload and production decision

One 4,096-output-token request was already decoding when three cold 65K
requests arrived. The following figures are one run per build:

| Metric | Baseline | Combined candidate | Kernel profiles, prior prefill policy |
|---|---:|---:|---:|
| Incoming requests' TTFT | 65.12 / 78.79 / 78.79 s | 63.25 / 74.74 / 74.74 s | 68.77 / 82.48 / 82.48 s |
| Existing request during incoming prefill | 1.41 tok/s | 1.07 tok/s | 1.19 tok/s |
| Largest progress gap during that phase | 2.94 s | 3.20 s | 3.37 s |
| Existing request when all four decode | 28.60 tok/s | 30.51 tok/s | 28.52 tok/s |
| Existing request's whole-run MTP acceptance | 62.48% | 61.94% | 56.91% |
| Entire mixed workload | 175.32 s | 173.59 s | 184.67 s |

The final incoming request reached first output about 5.1% sooner, but the
existing decode made 24.0% less progress per second while those prefills ran.
Its largest pause grew about 8.8%, for only a 1.0% reduction in total workload
time. There were no preemptions and all four output markers passed. These
single-run measurements identify a trade-off; they do not isolate each code
change's causal contribution.

The follow-up disabled only `prefill_remainder`, leaving the kernel profiles
enabled. It passed smoke, C4 cold/warm/control prefix and mixed-output checks,
with no preemptions, but did not confirm a recovery in serving performance.
Its total mixed-workload time was 5.3% longer than the baseline. The different
warmup sequence, output trajectories and one run per variant prevent attributing
all differences to kernel or scheduler changes. The larger allocation produced
29 progress events during incoming prefill; restoring the old policy produced
41, but the resulting token rates also depend on speculative acceptance.
For the existing request, whole-run mean acceptance length was 2.874 before,
2.858 with the combined candidate and 2.707 with the prior-policy kernel
variant. The latter needed 1,513 speculative steps versus 1,425 before to
generate the same 4,096-token count. This is an additional workload/acceptance
confound, not evidence that each GPU step became 5.3% slower.

Retain the previous validated baseline in production. The isolated kernel
gains and modest full-context combined result justify keeping the experiments
for further controlled work, but do not justify selecting either combination
as a dependable overall improvement. No CPU placement/thread change is adopted.
The example config reproduces the fully tested combined candidate; changing
`prefill_remainder` to `false` reproduces the follow-up variant.

The follow-up's safe numerical record is `gb10-sizing-kernel-variant.json`.
Both ranks again matched all 37 runtime module hashes and the BF16 plan hash.
That startup reported 1,418,906 logical KV tokens. Minimum available memory
during its checks was 13.35 GiB on the head and 16.71 GiB on the worker. After
readiness, approximately 239 seconds of sampling recorded 27.05/3.33 MiB of
swap-in and 0.035/0.008 MiB of swap-out on head/worker. These system-wide
counters show no sustained swap thrashing during the targeted tests; they
do not measure the complete NVMe mmap traffic. The full-context suite was not
repeated for this second variant.

## Correctness checks

The combined candidate passed API/authentication/tools/streaming smoke checks,
the scheduler and cache tests, 15 GPU QSA reference cases, output-length and
marker checks at C1–C4, and cold/warm C4 × 262K without preemption.

The initial free-format prefix test produced identical parsed JSON values but
different indentation (four spaces versus two) on the 12K cold/warm pair.
That exact-text check failed and is retained in the private evidence. A
follow-up with a fixed identity and an explicit compact-JSON output format
passed all six 12K/65K comparisons, including exact text and token sequences.
No model code was changed to make the follow-up pass.

An additional 12-request C4 check compared cold, warm and fresh-salt control
batches. All returned the expected identical JSON and token sequences. The
warm batch reused 38,400 prefix tokens; cold/control had zero prefix hits.
There were no preemptions. Across its eight paired comparisons, the largest
mean absolute token-logprob difference was 0.00415 and the largest individual
difference was 0.10547. These are bounded checks, not proof of identical
arbitrary outputs or repair-task quality.

## Mixed prefill/decode and the proposed GPU split

The scheduler reserves token positions for decoding, including speculative
inputs. It does not reserve SMs or a fraction of GPU execution time. The V2
runner executes the selected batch through one model forward; our full-decode
graphs cover uniform decode, while mixed batches use the eager path. Small
decode requests therefore share a model step with much larger prefill chunks.
Including decode tokens in every step does not guarantee short decode latency.

The suggestion to give decode 12–16 of GB10's 48 SMs and overlap prefill on the
remainder is an untested hypothesis. Bandwidth-bound kernels can still need
many SMs to keep enough memory transactions in flight. Prefill also uses DRAM,
L2 and execution resources. Lower decode power does not establish that 12–16
SMs can maintain its throughput, or that simultaneous prefill would leave its
bandwidth unchanged. The CPU/GPU link's bandwidth is not extra DRAM bandwidth.
Mixed matrix operations can also amortise weight reads across prefill and
decode rows; separating them may duplicate weight traffic. Any overlap gain
must exceed that cost as well as the cost of extra synchronization.

A useful next experiment would first measure representative decode kernels
under controlled launch concurrency, then measure their interference with
representative prefill GEMMs. Compare useful work completed and decode delay,
not power or a preferred SM fraction alone. Kernel-level overlap would need
a timeline confirming simultaneous execution. A server implementation would
also need separate scratch-buffer lifetimes and consistent TP/EP collective
ordering across both ranks. Ordinary CUDA streams alone do not promise a
fixed SM partition.

Changing prefill chunk limits is a smaller scheduling experiment: it trades
per-step overhead and prefill efficiency against decode responsiveness. That
is distinct from running prefill and decode concurrently on separate SMs.
No fixed GPU partition or concurrent model-forward path is implemented here.

Further serving comparisons should report target/draft step times and
speculative acceptance alongside generated tokens/s. The mixed follow-up shows
that equal prompt and output-token counts do not imply equal GPU work: it took
88 extra speculative steps for the long request. Matched warmup, cache capacity
and repeated alternating trials are needed before selecting a small gain;
power readings and one run per variant do not settle that choice.
