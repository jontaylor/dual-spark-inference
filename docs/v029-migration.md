# Official vLLM 0.29.0 migration — 2026-09-09

Status: validated and selected. Official 0.29.0 + retained Mia/GB10 work + the three-PR bundle + the replay-retention correction is serving.

## Comparison design

The baseline starts at upstream tag `v0.29.0`, commit
`98dff2a81d747d1dba01a47f939f48c3526d4206`, using the matching official ARM64
image pinned in the server Dockerfile. It retains selected Mia/GB10 changes
in separate commits. A second branch adds a focused PR bundle. Each candidate
gets one batched serving validation; a failing test is investigated before
adoption. Lightweight unit and kernel checks run before loading the model.

Both candidates use official Qwen FP8 weights, BF16 KV, TP2+EP+MTP3, C4,
262144 total tokens per request, 8192 scheduled tokens, full decode graphs,
Mamba align and semantic-only prefix retention. The endpoint, authentication
and complete target vocabulary remain host-configured as before.

The serving suite checks API/authentication, tools and streaming; cold versus
cached outputs and token log probabilities at 12K/65K; growing conversations;
matched warm C1/C2/C4 decode throughput; and cold/warm C4 at 262144 total tokens
with 1024 output tokens each. It records MTP acceptance, memory and preemptions.
Private prompts and responses are stored outside public Git.

## Retained work and release improvements

| Source | Included |
|---|---|
| Mia and the existing GB10 fork | Packed NVMe mmap PLE, FP8/NVFP4 row formats and original global-scale handling; native hash/gather; bounded row cache; mapped transport |
| Measured GB10 optimisations | BF16 GEMM plans, compact MoE projections/sparse activation, 98304-ID draft head with full target verification |
| GB10 cache/scheduling fixes | Reclaim sparse recurrent states across null holes; resolved prefill alignment; reserve in-flight recurrent-state copies; fair prefill and diagnostics |
| Official 0.29.0 | Model/loader refactoring, overlapping recurrent-copy fix, CUDA graph memory estimator, CUDA GDN/MTP path, batch-sharded sampler, per-request speculation metrics |
| Adapted #53899 infrastructure | Worker/executor/runner hooks required by the existing node-local mmap PLE path |

Inactive legacy ModelOpt and FP8-KV QSA overlays are omitted for this
official-FP8/BF16-KV experiment. The PLE hooks use model runner V2, explicitly
pinned in the launcher.

## PR bundle and deferrals

| PR | Disposition |
|---|---|
| [#53798](https://github.com/vllm-project/vllm/pull/53798) | Bundle: seed resumed recurrent state using the actual Mamba-group block size |
| [#56026](https://github.com/vllm-project/vllm/pull/56026) | Bundle: identify separately prefixed Qwen draft cache groups; adapt warning helper to the 0.29 API |
| [#55978](https://github.com/vllm-project/vllm/pull/55978) | Bundle: small metadata, dtype and no-op cleanup; expected gain may be modest under full graphs |
| [#54076](https://github.com/vllm-project/vllm/pull/54076) | Deferred: overlaps our geometry fix but also forces every state boundary to terminate a chunk, changing the benchmark's prefill policy |
| [#52244](https://github.com/vllm-project/vllm/pull/52244) | Deferred: broader fine-grained hybrid prefix/MTP checkpoint change; assess against reproduced failures |
| [#55260](https://github.com/vllm-project/vllm/pull/55260) | Deferred: stacked all-mode/GDN work partly targets another projection layout; 0.29 already supplies fused CUDA GDN/MTP |
| [#55122](https://github.com/vllm-project/vllm/pull/55122) | Deferred: native top-k rebuild and separate numerical validation needed |
| [#56028](https://github.com/vllm-project/vllm/pull/56028) | Deferred: repeated-preemption termination heuristic, rather than a capacity correction |
| [#54371](https://github.com/vllm-project/vllm/pull/54371) | Excluded: pins the complete CPU PLE shard, contrary to our reclaimable NVMe-backed memory objective |

The PR branch preserves individual commits so a regression can be isolated
without treating the whole bundle as one irreversible change.

## Validation record

Before serving: CPU cache simulations including C4 capacity/async lag/prefix
reuse; actual fair-prefill scheduler checks; native hash/gather and row-cache
differentials; draft vocabulary mapping/ties/scales; upstream and real FP8 mmap
PLE checks. Both nodes passed 212 graph replays in each PLE transport mode;
mapped transport left the CPU worker's CUDA context uninitialised. All 40 BF16
GEMM plans passed numerical checks. Full head-node MoE testing covered 64
pipeline cases, 96 changing-route graph replays and sparse-activation/fallback
checks; worker-node testing used a smaller differential suite.

The first model load exposed a stale draft-head plan symbol after the upstream
model rename. It was fixed, and the CPU test now exercises that optional import
path before a checkpoint load. No serving result was recorded from that attempt.

Nine targeted PR CPU regressions passed after adapting a helper absent from
0.29.0. The PR image also passed 16 upstream convolution gather/state-preservation
GPU cases and all five chunked verification-score checks.

The baseline passed API/authentication, arithmetic, tools and streaming. CUDA
graph allocation was 0.32 GiB against a 0.44 GiB reservation; the cache reported
1,409,197 logical tokens and 1,015 usable physical blocks. The 12K prefix check
failed: identical replay hit zero cached tokens, while a growing follow-up hit
9,600. All five returned the correct structured answer; identical and continued
outputs respectively matched their cold controls. The prefix assertion stopped
the suite before throughput/full-context tests. This candidate is not selected
for deployment. We proceeded to the prepared PR bundle rather than spending a
full capacity run on a candidate that had already failed a required check.

The initial PR bundle reproduced the same zero-hit identical replay (the
follow-up again hit 9600). That narrowed the problem to sparse recurrent-state
retention, rather than the group annotation alone. With our coarse 1600-token
lookup, scheduling wrote the checkpoint at 9600 for a 12K prompt, while
semantic-only retention selected 11200. The recurrent state needed by a replay
was consequently discarded. This also exposed a gap in the earlier CPU fixture:
it exercised dense retention, whereas production explicitly uses interval zero.

The follow-up fix shares the block-aligned replay-boundary calculation between
scheduling and Mamba retention. It applies the prompt-end exclusion before the
speculative tail rewind, and gives the Mamba manager the engine's speculative
mode even though that group itself does not drop a block. It preserves sparse
retention. This addresses the same producer/consumer-boundary principle as
#52244, but does not import that PR's broader fine-grained partial-tail changes.
Fine-grained lookup is outside the selected deployment and is unchanged.

The unmodified PR image fails the new 12K CPU reproducer (0 hits, expected 9600).
The corrected image passes 46 cache cases, including exact and adjacent block
boundaries, tiny prompts, 65K/261K, speculation off/on, and C4 with two prior batches pending (three including the current step). At C4/MTP3/261120 prompt tokens it hits 259200 per replay and peaks at
772 occupied blocks out of 995 usable. All other packaged CPU checks pass.
The GPU kernels and weights are unchanged by this correction.

The corrected bundle passes the full serving suite: API/auth/tools/streaming,
all cold/warm/follow-up comparisons at 12K and 65K, matched C1/C2/C4 throughput,
and cold plus warm C4 at 262144 total tokens per request. Warm hits are 9600 at
12K and 62400 at 65K. All six output comparisons return identical text; mean
absolute token-logprob differences are below 0.008, with a maximum below 0.333.
These are bounded correctness checks, not an end-to-end repair-quality result.

## Measured results

Each throughput mean uses the same ~12K source fixture, two engineering tasks,
two repetitions per task and 1024 generated tokens per request. Decode is timed
only while every request is producing output. Output trajectories can differ
between releases. The small differences do not establish a throughput gain.

| Concurrency | Previous validated build | Selected 0.29 build | Change | Draft acceptance |
|---|---:|---:|---:|---:|
| C1 | 48.47 | 47.43 | -2.15% | 62.7% |
| C2 | 75.85 | 76.08 | +0.31% | 61.8% |
| C4 | 115.71 | 115.27 | -0.38% | 63.1% |

Throughput is aggregate generated tokens/s. No throughput run preempted.

| Full-context C4 | Elapsed s | Median TTFT s | Decode tok/s | Peak sampled blocks | Preemptions |
|---|---:|---:|---:|---:|---:|
| Cold | 523.03 | 483.12 | 103.86 | 748/1042 | 0 |
| Warm replay | 44.27 | 4.24 | 108.45 | 728/1042 | 0 |

All eight full-context requests generated 1024 tokens and retrieved their embedded markers.
Minimum sampled MemAvailable during serving tests was 11.66 GiB on rank 0 and 16.71 GiB on rank 1.
The selected startup reported 1,446,646 logical KV tokens and 1042 usable physical blocks. Startup profiling varied across attempts; this is an observed capacity, not a fixed guarantee for every restart. C4 remains the configured limit.

The earlier validated C4 full-context run took 511.89 s cold and 41.94 s warm,
with 112.04/114.09 decode tok/s. Full-context decode is therefore **7.3% slower
cold and 4.9% slower warm** in this comparison. The cause has not been isolated.
The new run uses the same corpus and settings but independently generated marker
values; this is a workload comparison, not bitwise-identical output timing.

Warm replay reused 1,036,800 tokens total (259,200 per request), with a median
TTFT of 4.239 s versus 483.120 s cold. Both full-context decode intervals held
724 blocks throughout, with zero off-table blocks. Draft acceptance was about
63.3%, with a mean acceptance length of 2.897. The observations support successful
cache reuse and bounded allocation on this workload; they do not establish an
end-to-end quality or throughput improvement over the earlier deployment.

The original repair workload has not been rerun. Individual PR speed gains were not isolated; the release/PR combination was validated as a bundle.

## Selected source and images

Runtime source: `7e68a2093a416a9269380c050e79bdfcf5d5029d`. The exact 35-module manifest and aggregate results are preserved in the server provenance directory and `docs/v029-results.json`.

- Head image: `sha256:e49126e6768d0ef0edb39208080569af9408565d11afd55595aa7b33ff359ad1`
- Worker image: `sha256:230fdcc1d396ce64f9167c9208476b96baaa9f6832a837d4595b0a3a9cbff67d`

Images were built independently from the same source and pinned official ARM64 base; runtime Python module hashes match on both nodes. The source/deployment release tag is `gb10-v0.29.0-2026-09-09`. The original `main`/`gb10-2026-09-09` rollback remains intact.

## Rollback

Keep the original `main`/`gb10-2026-09-09` repositories and deployment directory.
Stop the head service (which stops the worker), point both service units back
to that checkout, reload systemd, and start the head. On the experiment hosts,
the candidate is selected by a separate `90-v029-candidate.conf` drop-in on each
node; removing only those drop-ins restores the original unit configuration.
