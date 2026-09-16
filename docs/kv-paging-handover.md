# GB10 same-process NVMe request paging

## Current production selection — original external PLE, 16 September 2026

At the user's request, both nodes now run the original **external-only PLE
implementation**, with one PLE process per node, CPU hashing, 16-thread mmap
gather and the original mapped output transport. The comparison wrapper,
inactive in-process reader, GPU hashing, policy mounts and custom syscall profile
have been removed. The verified pre-experiment runner/layer versions are restored;
other serving/paging settings remain s32/b8192/t1024, TP2/MTP3. Every inspected
process/thread remains unrestricted on CPUs 0–19.

Both configurations and source hashes passed dry-run validation before one
coordinated reload. Both hosts passed live verification and API health,
authentication, concurrent generation, tool-call and streaming checks. The
selection is based on maintainability: the comparison below did not demonstrate
a throughput winner. See [restoration and current operation](../experiments/ple-external-restore-20260916/README.md).

The repeating test client remains held idle. API/Prometheus and client token
timings remain available; the dual-backend per-step logger is no longer loaded,
and the obsolete native-only monitor remains stopped. Backend policy files are
no longer mounted, so changing them has no serving effect. A future dual-backend
comparison needs an explicit redeployment of its archived configuration.
Earlier current-state entries below describe historical experiment windows.

## Current PLE comparison outcome — 16 September 2026, 14:33 BST

Both full PLE backends are deployed behind a runtime selector, sharing the same
captured output/flag addresses. The original external worker path retains its
ZeroMQ handoff, CPU hash and 16-thread gather. The in-process path retains GPU
hashing, SQPOLL and FULL-graph overlap. Exactly one path processes each batch;
each retains its own cache in the comparison deployment. Loaded sources are
frozen in `experiments/ple-backend-comparison-20260916/`.

The completed fixed c12 campaign found **no demonstrated throughput winner**:
external 261.46 versus in-process 262.57 output tokens/s. The four-pair geometric
difference was +0.42%, with a preliminary 95% interval of −0.78% to +1.65%.
All eight measured cycles did the same client/server token work; all 120
requests including warmups passed repeat-output checks. Both nodes remained
healthy with unchanged processes, and 5,851 aligned completed steps passed the
telemetry audit. This is a warm synthetic workload, not a cold-I/O or general
quality result. See [final analysis](../experiments/ple-backend-comparison-20260916/FINAL_ANALYSIS.md).

**Current selection is in_process, epoch 14**, on both nodes. The new uint64
`ple-backend-policy.bin` controls the full backend. The independent uint32
`ple-read-policy.bin` remains zero, selecting SQPOLL and disabling the rejected
prefetch candidate. Do not confuse these controls. Server settings remain
s32/b8192/t1024, TP2/MTP3; all process/thread affinities are unrestricted 0–19.
The external worker processes remain resident to allow switching without a
restart. Do not infer the selected backend from process presence.

The remote repeating client on jon@192.168.0.167 is held idle after ten cycles.
Coordination uses the user-authorized thread **Review latest planner experiments (3)**,
ID `01a09b9f-c6c7-7bb0-8c25-b22748f43388`. Common backend-aware JSONL telemetry
remains active on both ranks. The previous native-only ple-serving-monitor
service is stopped intentionally; it cannot attribute switching intervals.
Switch only with the client held and serving drained, using the experiment's
`set_backend.py`; its README documents verification and rollback. No additional
restart is needed. Do not overwrite frozen mounts or daemon-reload unrelated
pending unit edits. Earlier sections below are historical.

## Latest PLE decision — 16 September 2026, 13:03 BST

The batched-prefetch candidate was rejected after a same-worker/cache live
comparison: GPU readiness regressed by0.44–0.63ms in matching10/12-request
cycles on both ranks. Whole-step timing had only one matching cycle and did
not establish superiority. Both hosts' explicit PLE control files are fixed to
**mode0, the previous SQPOLL read path**; no ABBA switching remains. The running
binary is the tested experimental reader in `experiments/ple-resident-20260916/`,
not the previous binary, and canonical source remains reader17. Full binary
rollback configs and launchers are preserved in that experiment directory.
EngineCore/GPU workers remain unrestricted on CPUs0–19; serving remains
s32/b8192/t1024,TP2/MTP3. Current decision and evidence are recorded in
[selection](../results/ple-resident-20260916/selection.json) and
[experiment notes](../experiments/ple-resident-20260916/README.md).
Earlier status sections below describe previous experiment windows.


## Current PLE status — 16 September 2026, 07:35 BST

Fixed SQPOLL is now live on both ranks: native/connector17, GPU hash/wait12,
overlap runner06. API smoke and live mount/environment/parameter verification
passed. The completed 20-minute stability capture passed with 5,880 aligned
batches, zero native errors and zero rank shape/policy mismatches. Both containers
remained healthy with no restarts. Configuration is restored to **s32/b8192/t1024, TP2, MTP3**. PLE stays
inside the GPU worker, with all unique misses submitted before waiting and
without per-step application IPC or forced IOSQE_ASYNC. Experimental ABBA
switching is disabled.

The controlled 12-request comparison favoured SQPOLL in all 14 cycles on both
ranks: GPU hash-to-ready-gate improvement 0.244/0.224 ms. At 24–25 requests the
improvement was approximately 0.75–0.81 ms. Whole-step confidence intervals
still include zero. These timings do not certify GPU L2 residency or model
quality. See [campaign report](../experiments/ple-latency-campaign-20260916/FINAL_REPORT.md)
for evidence, limits, validation and rollback.

GPU worker threads have all CPUs 0–19. At the user’s subsequent request on
16 September, EngineCore 127306 was also unpinned to CPUs 0–19 for all 76
threads, without a restart. A short before/after check found no clear matched
batch speedup; see [affinity comparison](../results/ple-engine-unpin-20260916/REPORT.md). Both NVMe
latency constraints are 0, persisted with the campaign's udev rule (higher idle
SSD power; no reboot test). GPU-validated rollback configs are
`experiments/ple-latency-campaign-20260916/deploy.fixed-normal-r{0,1}.json`.
Install BOTH host configs before head start, because its ExecStartPre starts
the remote worker. Do not overwrite frozen mounted assets or daemon-reload
unrelated pending unit changes. Historical campaign entries below are superseded
by this current-state section.


This deployment adds reservation-based admission and active-request parking to the
existing NVIDIA NVFP4 service. It keeps TP2, expert parallelism, MTP3, the 98,304-ID
draft vocabulary, BF16 KV, RoCEnante and the packed FP8 PLE table. The current
PLE reader is described in the status section above.

## Serving configuration

- Endpoint: `http://spark-1:30001/v1`; existing API key and served aliases.
- Maximum context: **262,144 tokens**, including prompt and generation.
- Maximum resident scheduler requests: **32**. Large requests can queue before
  this ceiling because admission uses actual physical cache reservations.
- KV pool: **40 GiB per node**; NVMe cache budget: **48 GiB per node**.
- Reservation growth: **32,768 tokens**; fixed recurrent-state and speculative
  allowances are charged separately. The measured budget is 1,880 blocks,
  with 21 blocks per token unit and 26 fixed blocks per request (1,897 physical
  blocks including allocator headroom). Thus 32 small requests reserve 1,504
  blocks; nine eight-unit requests reserve 1,746. This is reservation arithmetic,
  not a measured nine-way full-context workload. A near-boundary speculative
  allowance can move a request into its next unit. The raw vLLM capacity report
  is 2,688,038 tokens / 10.25 full contexts, before this conservative admission.
- Local NVMe directory: `/home/jon/.cache/vllm-gb10-kv-paging` on each node.
- Four staging blocks; transfer hashing and GPU readback enabled for this initial
  handover. This diagnostic work has overhead; no throughput claim is made.

The queue is FIFO. A parked request returns to its front, and must reserve its
next growth unit before restoration. A large head request can hold up smaller
requests. All ranks must complete saving before GPU blocks are released, and
all ranks must complete loading before the request is promoted. At an aligned
checkpoint, replay is bounded to 3,200 tokens. Requests paused before parking
retry growth when another request frees capacity.

## Operation

The historical systemd unit names still contain `fp8`; the loaded weights are
**nvidia/Qwen3.8-Flash-Next-NVFP4**.

On spark-1:

```bash
sudo systemctl start qwen38-next-qwen-fp8.service
sudo systemctl stop qwen38-next-qwen-fp8.service
sudo systemctl restart qwen38-next-qwen-fp8.service
journalctl -fu qwen38-next-qwen-fp8.service
curl -fsS http://127.0.0.1:30001/health
```

The head starts/stops `qwen38-next-qwen-fp8-worker.service` through SSH. Containers
are `qwen38-kv-paging-r0` and `qwen38-kv-paging-r1`. Inspect scheduler events with
`docker logs qwen38-kv-paging-r0 2>&1 | rg 'GB10_PARKING|GB10_KV_ACCOUNTING'`.
Prometheus metrics are at `/metrics`: running/waiting requests, KV usage,
generated tokens, speculative acceptance and unexpected preemptions.

Restart is deliberately manual after failure (`Restart=no`). Missing/corrupted
checkpoint files, unavailable bounded checkpoints or accounting failures stop
serving instead of silently recomputing. Inspect logs and disk/memory first,
then restart the head unit to restart both ranks. Restart drops queued and
active requests. Clients must handle errors and retry.

Startup validates the pinned image and overlay hashes, removes only expired
UUID namespaces inside the marked dedicated cache directory, and requires the
configured disk budget plus 8 GiB free. It refuses cleanup while a known rank
container is running. The supervisor stops on sustained low host memory.

## Rollback

```bash
sudo systemctl stop qwen38-next-qwen-fp8.service
sudo rm /etc/systemd/system/qwen38-next-qwen-fp8.service.d/zz-paging-handover.conf
ssh 192.168.100.11 'sudo rm /etc/systemd/system/qwen38-next-qwen-fp8-worker.service.d/zz-paging-handover.conf && sudo systemctl daemon-reload'
sudo systemctl daemon-reload
sudo systemctl start qwen38-next-qwen-fp8.service
```

Only the new override is removed. Earlier NVFP4 service overrides remain in
place, so rollback restores the outgoing configuration.

## Source and reproducibility

vLLM worktree: `/home/jon/vllm-gb10-kv-paging`, branch `gb10-kv-paging`, commit
`fd57262a959793b46565b56a116f7e6a5e1a34c6`. Deployment worktree:
`/home/jon/dual-spark-inference-kv-paging`, branch `gb10-kv-paging-handover`.
No GitHub push is part of this handover.

`paging-manifest.json` pins six source overlays and two checkpoint config files.
`prepare_paging.py --server-root /home/jon/vllm-gb10-kv-paging` rebuilds the bundle
only from committed source. `deploy_config.json` and model artifacts are local,
gitignored runtime state; the key remains in its existing separate file.
The image IDs are different on the two nodes and individually pinned.

## Validation and limits

- CPU policy/lifecycle/adapter/transport suite: 28 passed; three GPU tests skipped
  there and exercised separately. Disk/DMA suite: six passed on each actual GPU.
- Live four-request pressure run: four 95,032-token prompts generated 4,096 tokens
  each; one park/restore with a decode-created 96,000-token checkpoint and
  2,301-token replay. Both ranks verified restored bytes. A growth retry bug
  found during testing was fixed and regression-tested.
- Live cancellation during restore: cancelled request never promoted; three
  peers completed and a subsequent probe succeeded.
- Missing worker checkpoint files: explicit FileNotFoundError, affected clients
  received errors, no restore promotion and health failed. No silent prefill.
- Numerical controls support continuity at sampled boundaries, not bit-exact
  equivalence. Two cold controls matched 16 resumed tokens; a third diverged
  after 12. A small probability difference exceeded the cold-control spread.

This is an experimental, text-only, synchronous scheduler for this hybrid model.
Metadata is **same-process only**: disk files cannot restore a restarted engine.
Completed-conversation retention and guaranteed 15-minute follow-up reuse are
not delivered. Buffered writes do not promise power-loss durability. Extended
soak testing and broad numerical equivalence testing remain future work.

Detailed artifacts and failed attempts are preserved at
`/home/jon/gb10-optimisation/20260910-kv-paging-prototype/`.
Final serving checks are recorded in `results/paging-handover/`.

For a sampled status report (aggregate generation rate, concurrency, cache,
draft acceptance and both GPU power readings), run from the deployment checkout:

```bash
.venv/bin/python tools/paging_status.py --interval 10
```

GPU utilization/power are instantaneous end-of-window samples; throughput and
acceptance are counter deltas over the interval. Power does not measure memory
bandwidth. A parking event can contribute to the base preemption counter;
use the `GB10_PARKING` events to distinguish planned parking.

The bundle depends on the already-tested local images and model/PLE artifacts;
it is not a fresh-machine image build recipe. The old generic preparation
scripts are retained as historical tooling. Follow this handover guide for
this branch. No source or deployment changes have been pushed to GitHub.

## Final serving acceptance, 2026-09-11

The normal endpoint passed with the final 40GiB-per-node configuration:

- All 24 simultaneous short requests completed 256 outputs each; sampled
  running concurrency reached 24. These are synthetic acceptance requests,
  not a representative throughput benchmark.
- A 262,016-token prompt completed 128 outputs (262,144 total) in 114.23 seconds.
- Zero preemptions during these checks; health remained successful.
- Missing and incorrect keys were rejected. Chat returned `ready` through the
  existing `ornith-1.5-35b-a3b-tp2` alias and original key.
- Both nodes retained approximately 20GiB available host memory during the
  long-context check. Existing host swap remains allocated; short vmstat
  samples showed no sustained heavy swap I/O. This is not a long-term soak.

The server is left running with both systemd services active. The exact
aggregate evidence is in `docs/kv-paging-validation.json`; full request and
metric records remain local in `results/paging-handover/`.

## Cached-token reporting and C32 update

The current configuration sets `max_num_seqs=32` and enables
`--enable-prompt-tokens-details`. MTP3 graph capture sizes extend to128 tokens
for32 sequences. Context and40GiB KV/48GiB disk budgets are unchanged.
The original C24 acceptance results above remain historical evidence.

Chat/completion responses include `usage.prompt_tokens_details.cached_tokens`.
For streamed responses, send `stream_options: {"include_usage": true}` and
read the final usage chunk. Uncached prompt tokens are prompt_tokens minus
cached_tokens; sum these values for a total and take their maximum for the
largest uncached prompt. Missing/null cached counts mean unknown, not zero.
The field measures initial prefix reuse, not additional parking replay work.

Live verification on11September:32 simultaneous requests each completed256
outputs, with zero preemptions. An11254-token chat prompt reported0cached
tokens cold and9600cached tokens on repetition, in both ordinary and streamed
responses (1654uncached). Evidence: `docs/c32-cached-token-validation.json`.
The service is left running with the original port/key and262144context limit.

## SSE keepalives

The launcher enables `--sse-keep-alive-interval 10`. Chat/completion SSE streams
emit `: keep-alive` comments during idle periods. SSE-aware consumers should
ignore comments as content while allowing their arrival to reset network read
inactivity timers. This does not turn non-streaming requests into streams or
extend an absolute proxy/client deadline.

Live verification: a cold102400-token streamed completion emitted keepalive
comments at10.02,20.02 and30.02seconds, then completed successfully at39.96s
with cached-token usage details. Evidence: `docs/sse-keepalive-validation.json`.

## Exact completion checkpoints

`kv_paging.completion_checkpoints=true` adds a completion-only hybrid snapshot
between aligned boundaries. Matching follow-ups can reuse all but the last
unprocessed token, while active-request parking remains aligned. See
[completion-checkpoints.md](completion-checkpoints.md) for state coverage,
validation, eviction, and rollback details.

## Measured scheduler recommendation applied, 2026-09-15

Following the user's request to implement the final work-score recommendation,
both nodes' `deploy_config.json` now use `max_num_seqs=32`,
`max_num_batched_tokens=8192`, and `long_prefill_token_threshold=1024`.
Only the token budget changed from the preceding live configuration (16384).
The example configuration now explicitly includes the 1024 threshold.

The source is [the final campaign analysis](../experiments/empirical-work-score/smooth-final/FINAL_ANALYSIS.md):
this is the best observed computational-work configuration, not a proven optimum
or a correctness certification. Existing supplementary quality failures remain.

Both launchers passed dry-run validation and both restarted containers expose
the intended settings. Evidence is in `results/scheduler-recommendation-20260915/`.
Each node retains its previous configuration at
`deploy_config.json.before-work-leader-20260915`; restoring those files and
restarting the head service rolls back the budget change.

Post-restart API validation passed: health 200, unauthenticated access rejected
with 401, both model aliases present, four concurrent arithmetic responses
correct, weather tool call correct, and streamed `READY` completed. Results are
in `results/scheduler-recommendation-20260915/smoke.json`. These are smoke checks,
not a repetition of the campaign or broad quality certification.

## In-process PLE lookup, 2026-09-16

Both nodes now enable `optimizations.ple_in_process`. Hashing and batch read
submission execute inside the GPU worker, with no separate PLE process or
per-step PLE ZeroMQ handoff. All unique row-cache misses are submitted through
`io_uring` before waiting; output is published only after all reads succeed.
The container seccomp profile permits the required io_uring syscalls.

Health, concurrent generation, tool calls, streaming and authentication checks
passed after the coordinated restart. See
[implementation and rollback](../experiments/ple-in-process-20260916/IMPLEMENTATION.md)
for constraints, evidence and previous config/launcher copies. The scheduler
remains s32/b8192/t1024. EngineCore affinity is 15–19, GPU workers 0–19.
No end-to-end speedup or broad quality certification is claimed.

### PLE reader regression investigation, 2026-09-16

The initial in-process reader forced IOSQE_ASYNC on software-cache misses. At 768 rows a spark-1 profile measured 3.608 ms median native time, 3.125 ms median poll residence and 44.5 polls/batch. A candidate in `vllm-gb10-kv-paging/gb10/ple_batch_reader.c` removes forced worker scheduling and uses a bounded batch-completion wait after all submissions. Matched warm-file/no-row-cache 768-row median improved from 0.863 to 0.185 ms; this is a standalone benchmark, not a live result. Native tests and real GPU integration passed. **Historical status at that investigation:** the candidate was not yet deployed. It has since been superseded by the latency campaign below. Full evidence, limitations, candidate binary/source, and preserved baseline are in `results/ple-reader-investigation-20260916/REPORT.md`. Direct mmap copies still win some native microbenchmarks; do not claim a universal crossover or end-to-end gain.


### Latency campaign history, 2026-09-16

The forced-worker fix is now deployed, together with direct cache-to-output
copies, GPU-side exact row-ID hashing and disabled NVMe autonomous low-power
transitions. A spark-1 on/off/on experiment measured roughly 1.77 → 0.87 → 1.81 ms
at 1536 rows, with spark-2 serving as the unchanged comparison; both controllers
now have `power/pm_qos_latency_tolerance_us=0` (original 100000). This policy increases idle SSD power. It is now reapplied by the verified
`/etc/udev/rules.d/99-gb10-ple-nvme-latency.rules` on both nodes when nvme0
is added or changed; device-event tests passed, but a host reboot was not performed.

At 04:19 BST both services began loading candidate 06, which submits reads before
full CUDA graph replay and completes them on the same CPU thread after launch.
Eager/piecewise execution stays synchronous. This restores potential overlap
without any PLE subprocess or message bus. Exact native tests, CPU/CUDA-input real graph integration and live API smoke
passed. Both nodes are healthy with automatic load resumed. The live timeline
confirms overlap; occasional late publication under profiling remains under
investigation, so no controlled end-to-end gain is claimed. The immediately
preceding healthy configuration is `deploy.gpu-hash-r{0,1}.json` in the campaign
folder; restore the corresponding config on each host and restart to roll back.

Current configuration, evidence and remaining work are recorded in
[the campaign worklog](../experiments/ple-latency-campaign-20260916/WORKLOG.md).
Do not infer the active variant from the development checkout: serving mounts
frozen per-variant files with verified hashes. No end-to-end gain is yet certified.


At 04:43 BST, both nodes passed smoke with the same overlap path plus candidate
07 GPU wait telemetry. Nsight is disabled. The verified optional
`ple_mapped_wait_library` records wait duration and poll count in spare status
buffer words; ready, delayed-publication and timeout behavior passed GPU tests.
Active configs are `deploy.telemetry-r{0,1}.json`. No async issue-policy experiment
is enabled in these configs; candidates 08/09 remain experimental.

Low-overhead acceptance measured 1,072 completed GPU wait samples across both
ranks, including 24-/32-request decode batches, with no polling for late PLE
data and no native errors. Wait-loop ready checks were at most 0.512 µs; these
exclude kernel launch/prologue cost. See `telemetry-baseline` and
`telemetry-steady` results. Candidate 08 A/B configs are staged only and await
GPU validation before any rollout.


At 04:57 BST candidate 08 passed GPU validation and live smoke and began an
ABBA submission-policy experiment (`deploy.async-ab-r{0,1}.json`, 64-step blocks).
This is an active experiment, not the final chosen policy. Ten-minute capture
found faster asynchronous submission but occasional millisecond GPU waits;
end-to-end evidence remains inconclusive because few complete cycles matched
request counts. Candidate 07 remains the healthy rollback. The campaign worklog
tracks the active CPU-placement capture and next controlled affinity test.


NVMe policy rollback: remove that udev rule on both nodes, reload udev rules,
and restore `/sys/class/nvme/nvme0/power/pm_qos_latency_tolerance_us` to `100000`.
The source rule and device-event validation logs are retained in the campaign.

At 06:02 BST candidate12 GPU-clock timing passed live acceptance on both ranks:
all configured runtime mounts verified, API smoke passed and representative load
resumed. Active configs are `deploy.gate-timing-r{0,1}.json`; native reader08,
connector/hash/wait12, runner06. The64-step ABBA normal/forced-worker experiment
remains enabled and is not the final selected policy. GPU workers and their
threads are unrestricted0–19; EngineCore15–19. Candidate16 SQPOLL comparison is
staged only and still requires GPU integration before deployment.

Operational correction: the head unit's **ExecStartPre starts the remote worker**.
Install and verify BOTH host configs before starting the head; otherwise the
remote rank can load the previous config. The first12 restart hit this race and
was corrected. `verify_live.py --config-prefix deploy.gate-timing --output PATH`
now checks every live runtime mount on both nodes immediately after startup.
The mismatched first capture is retained as `gate-timing-validation` and must
not be used as a two-rank12 comparison. Use `gate-timing-corrected` instead.

At 06:10 BST candidate16 passed GPU validation and began loading for a
same-reader/cache **normal-versus-SQPOLL** ABBA comparison. Active configs are
now `deploy.sqpoll-ab-r{0,1}.json`, verified on both ranks immediately after
startup. Reader16 uses the `submit_async` experiment entrypoint for SQPOLL;
it does **not** force IOSQE_ASYNC. Capture metadata must say
`--async-policy sqpoll`. Connector/hash/wait12 and runner06 are unchanged.
Live smoke and the first600s timing capture remain pending; see WORKLOG.md
for the active job. This is still an experiment, not the final selected policy.
