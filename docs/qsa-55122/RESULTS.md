# QSA #55122 backport — execution record

Backport deployed and retained on both Sparks. The API is healthy, both live
library hashes match the validated artifact, and all 11 serving marker probes
passed with normal stop completions. The representative workload is running
with 10 active trials and has resumed successful responses.

## Delivered change

- Upstream revision: `42db1ddbbd8916bfde6b002e195a7e0ef35b0450` from
  https://github.com/vllm-project/vllm/pull/55122.
- Native source base: `98dff2a81d747d1dba01a47f939f48c3526d4206`, matching the
  installed v0.29 wheel. Isolated checkout:
  `/home/jon/vllm-gb10-qsa-55122`, branch `gb10/qsa-deterministic-55122`.
- Backported kernel header, launcher and upstream tests. The launcher adopts
  the PR's deterministic low-shared-memory fallback. Header differences from
  the pinned upstream version are clang-format only.
- Added an opt-in CMake switch to omit unchanged external packages when building
  the native library. No unrelated native source change or Python model change.
- Native library SHA256:
  `1687b153cda9a88293abbe06ceb72b335df29372e2cf9f5cd6567539019ba491`.
- Candidate tag on each node: `vllm-gb10:v029-roce-qsa55122`. Each node retains
  its own validated image base and receives the identical patched binary.
  Exact old/new image IDs are in `image-identities.json`.
- Deployment edits change only `image` and `image_ids` in each node's existing
  configuration. Paging, FLA patch, MTP, caches and sampling are preserved.

## Validation

| Check | Stock | Candidate |
| --- | --- | --- |
| Upstream persistent-top-k tests, Spark 1 | 87 passed, 134 failed, 26 skipped | 221 passed, 26 skipped |
| Adjacent large-vocabulary decode top-k tests | Not run | 2 passed |
| Graph, paged QSA, empty-batch and activation checks, Spark 1 | Not run | 11 passed |
| Upstream persistent-top-k tests, Spark 2 | Not repeated | 221 passed, 26 skipped |
| Graph/paged-QSA checks, Spark 2 | Not repeated | 11 passed |
| NVFP4 quantization and GEMM regressions, Spark 1 | 72 passed | 72 passed |
| Narrow-score measurement grid | All 8 cases non-repeatable | All 8 exact and repeatable |

The stock narrow-score 128-row / 163840-pitch / 8192-active / k=512 case selected
15775 values below the true kth score. The candidate selected none below the
threshold in any measured case. This is synthetic candidate-loss evidence,
not a count of errors in production requests.

Graph checks include changing score values and ragged lengths across replays,
both sides of the 22016 transition, capacity-sized strides, and a 400000-column
case exercising the low-memory fallback on GB10. Paged-QSA tests use the deployed
scorer and expander with permuted physical pages and multiple requests.
The largest possible row-state allocation on this 48-SM/1024-thread configuration
is 172224 bytes, within the caller's 1 MiB workspace.

The first custom harness run had three harness errors: its extreme-width probe
exceeded the PR's 64-CTA limit, and two activation probes used a removed Python
wrapper. The corrected tests use a width that reaches the intended fallback
and the public `torch.ops._C.silu_and_mul` operator. No kernel change was made
in response, and all corrected tests passed on both GPUs.

Ruff checks, upstream-test formatting, clang-format checks and `git diff --check`
passed. Full logs and `native-provenance.json` are retained beside this report.

## Performance and scope

The eight measured kernel cells range from 0.507x to 1.352x stock latency under
the existing background GPU load. This is not a universal speedup, and the
baseline is stock persistent_topk, not the slower torch.topk workaround.

Before deployment, all 11 serving marker probes passed. The three cold 20K
prompts had median TTFT 9.109 seconds; the three changed-suffix prefix reuses
had median 1.213 seconds. The server reported cached tokens for reused prompts.
This prefix-cache benefit already existed before the backport.

After startup (455 seconds from readiness monitoring), the agent sessions had
to rebuild their prefix caches. An initial timing attempt during that recovery
was interrupted and excluded. Measurement resumed after accounting showed all
10 active sessions in decode, with no prefill tokens remaining.

| Serving case | Stock median TTFT | Candidate median TTFT | Requests per arm |
| --- | --- | --- | --- |
| Cold 20K | 9.109 s | 7.653 s | 3 |
| Reused 20K, changed suffix | 1.213 s | 1.167 s | 3 |
| Cold 32K | 11.983 s | 13.812 s | 1 |
| Four concurrent 8K prompts | 12.289 s | 14.012 s | 4 |

Both arms returned 17600 cached prompt tokens for each reused 20K prompt and
zero for each cold prompt. All 11 markers passed in each arm.

Timings were mixed: 20K TTFT improved in this sample, while 32K and concurrent
8K TTFT were about 15% and 14% higher. Per-request decode medians were also
lower in the candidate sample (for example, 19.07 to 11.94 tok/s for the cold
20K cases). These were sequential measurements with changing background agent
traffic and different cache histories, not a controlled isolated A/B. They do
not establish either a universal speedup or an attributable serving slowdown.
The candidate is retained for its verified correctness fix; no claim of zero
performance cost is made. Full request data and summaries are in
`serving-comparison.json`, `stock-serving.jsonl` and
`candidate-steady-serving.jsonl`.

This is a GB10-specific native build. The operator inventory omits two
architecture-specific registrations present in the multi-architecture wheel:
Kimi K3 AttnRes (requires SM100) and the MXFP4 experts support-query registration.
The supported Python query returns False for SM121 in both stock and candidate;
these omissions do not change supported Qwen/GB10 functionality. This artifact
should not be used as a general multi-GPU-architecture wheel replacement.

The PR remains open. Top-k correctness and repeatability do not establish full
generation determinism. The inherited inter-CTA spin barrier remains in the
large-row path; tests cover our shapes but do not prove every concurrency regime.

## Build and rollback

The first two-job build exceeded a 12 GiB builder cap in an unchanged FP8
translation unit. A one-job incremental continuation completed successfully.
The serving process was preserved during compilation. The user subsequently
clarified that coordinated inference downtime should be used to reclaim RAM
and enable faster parallel full builds. Follow that preference for future full
rebuilds; `build-native.sh` accepts `QSA_BUILD_JOBS`.

The build environment and all successful objects are retained for incremental
rebuilds. `backport.patch`, `build-native.sh`, `Dockerfile.runtime`, CUTLASS v4.4.2
archive/hash, test sources and provenance are retained in this directory.

Rollback changes only image identity with `set_image_arm.py stock` and the
saved `image-identities.json`, then restarts both ranks. Coordinate with the
representative-load task before a restart. No running process should have its
library replaced in place.
