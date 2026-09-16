Complete PLE backend comparison, 16 September 2026.

**Subsequent production decision:** the user selected the original external
workers for maintainability. The external-only restoration removes the dual
backend runtime and its inactive native reader; see
[the restoration record](../ple-external-restore-20260916/README.md).
The switching instructions below apply to this archived comparison deployment,
not to the external-only production configuration.

**Completed at 13:33 UTC.** Four measured cycles per backend found external
261.46 output tokens/s versus in-process 262.57, a paired difference of +0.42%
(preliminary 95% interval −0.78% to +1.65%). No throughput winner was established.
Both hosts remain healthy in fixed in_process mode, epoch 14; the client is
held idle. All 120 requests passed the final repeat-identity audit. See
[FINAL_ANALYSIS.md](FINAL_ANALYSIS.md) for results and limits, and [CURRENT.md](CURRENT.md)
for the operational handover. The instructions below describe the completed
deployment; do not rerun campaign.py against its existing evidence directory.

The deployment retains the original CPU worker, connector, protocol, native
CPU hash and 16-thread mmap gather. Their source hashes match the previously
deployed image. The second backend is the current in-process GPU-hash/SQPOLL
reader with FULL-graph overlap and synchronous eager gathering. The rejected
page-prefetch algorithm remains disabled in the independent read-policy file.

`PleBackendComparison` gives both producers the original connector's shared
CPU output allocation, CUDA mapping, completion flag and consumer event. The
model's captured addresses never change. Exactly one backend receives each
real batch; there is no shadow lookup. Completion sequence numbers remain
monotonic across switches. Native constructor changes only allow borrowed
transport storage; native read/hash algorithms are unchanged. The original
external path retains its input staging, ZMQ request and CPU hash/gather work.

Both backends retain their own original row caches. This adds the inactive
cache's memory footprint; both arms share this deployment footprint and the OS
file cache. Each arm requires warmup. Results describe this paired deployment;
they do not establish cold-start performance or the memory-pressure difference
between independently deployed single-backend services. OpenMP remains PASSIVE,
CPU affinity remains unrestricted, and s32/b8192/t1024, TP2/MTP3 are unchanged.

The control is `/home/jon/.cache/vllm-ple-control/ple-backend-policy.bin`, mounted
at `/opt/gb10/ple-backend-policy.bin` on each host. It is 4096 bytes with one
aligned atomic uint64: bits 0–7 mode, 8–23 block length, 24–31 reserved zero,
32–63 epoch. Modes are 0=in_process, 1=external, 2=ABBA, 3=BAAB. ABBA/BAAB use
the common real batch sequence, never GPU capture iterations or wall clocks.
Fixed modes are used for the repeating client comparison. No added per-step
network synchronization is required. Both ranks' records must agree before a
comparison is accepted.

While client load is held and the server is drained:

```
.venv/bin/python experiments/ple-backend-comparison-20260916/set_backend.py \
  --mode external --label cycle-label
```

Use `--mode in_process` to return to the previous reader. Omitting `--mode`
reads/verifies both host control words. The tool requires two idle API metric
observations, atomically updates and verifies both host files, and records the
epoch boundary in `results/ple-backend-comparison-20260916/control.jsonl`.
The workload controller must remain held until it returns successfully. Host
updates are sequential, so this is not an atomic distributed switch under load.

Each GPU worker writes `live/rankN/rankN-BOOT.jsonl` on its own host. A `begin`
record identifies backend/epoch/sequence; a delayed `step` record contains:

- CUDA elapsed time from before real input preparation through model forward,
  sampling, drafting and post-forward enqueues, plus the forward subinterval.
- CPU prepare/completion/fence/enqueue durations.
- Existing GPU wait-loop body time, polls and error flag from the preceding
  consumed batch, associated by its sequence number.
- Actual request/token/padded shape, prefill mask, scheduled draft tokens,
  per-request context upper bounds, and accepted draft-token count.
- Sampled-token count before API stop filtering. This is not the headline
  useful-output count; client completion-token usage provides that.

CUDA events are resolved only after querying completion; there is no new
per-step synchronization for timing. The original consumer fence is shared and
performed once. Log writes/event recording have overhead and are identical in
both arms; no overhead-free claim. Results cover preparation through drafting,
not network delivery or engine scheduling outside that interval. The final
step can remain buffered until the next request or clean close; `begin` records
remain visible immediately. The last batch's wait data can be absent.

The prior uprobe monitor is stopped during this experiment because it only
recognizes native-reader batches and would misattribute intervals across the
external arm. Client/API metrics and the common backend telemetry replace it.

`check_backends.py` passed 240 actual GPU-consumed exact-byte checks using the
real worker registration, ZMQ dispatch, CPU hash/gather and publication code
with a small synthetic table. It switches both fixed and alternating modes
with CPU and CUDA inputs, eager/FULL execution, padded request shapes, EOS,
negative speculative placeholders, monotonic sequence numbers and delayed
external completion. Both producers retain the same output pointer across
graph replay. This validates transport/computation plumbing, not model quality.
Evidence: `results/ple-backend-comparison-20260916/integration.log`.

The repeating load is owned by remote thread
`01a09b9f-c6c7-7bb0-8c25-b22748f43388`, title
`Review latest planner experiments (3)`, on `jon@192.168.0.167`. Coordination
status is `/tmp/ple-repeating-load-status.json`. The prepared c12 load has fixed
prompts/settings, 12 requests per cycle and 2048 maximum output tokens each.
Two warmup cycles precede eight measured cycles, ordered external/native/native/
external then native/external/external/native. Every cycle drains before the
next backend epoch. Four adjacent opposite-backend pairs support a preliminary
paired throughput estimate; uncertainty is at the cycle level, not per-token
pseudoreplication. The workload is synthetic and cannot certify agent quality.

Initial deployment starts in in_process mode. Before admitting measured load:
verify both hosts' source hashes, runtime mounts, control words, unrestricted
affinity, health, and functional requests through both paths. Never edit a
loaded frozen source or library. A full rollback uses `deploy.before-r0.json`
and `deploy.before-r1.json`, installing BOTH configs before head start. No
systemd daemon-reload is required or permitted for this experiment's rollout.
