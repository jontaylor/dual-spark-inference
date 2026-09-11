# GB10 same-process NVMe request paging

This deployment adds reservation-based admission and active-request parking to the
existing NVIDIA NVFP4 service. It keeps TP2, expert parallelism, MTP3, the 98,304-ID
draft vocabulary, BF16 KV, RoCEnante and Mia's FP8 PLE mmap path.

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
