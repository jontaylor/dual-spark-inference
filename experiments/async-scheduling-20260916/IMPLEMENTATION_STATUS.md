> Shelved at the user's request on 16 September 2026.
> Branch: `experiments/async-scheduling-20260916`. No async deployment or profiling restart is authorized by this artifact.
> Resume only on a new user instruction; validation gates below remain open.

# Async implementation checkpoint

The goal remains incomplete. No async candidate is deployed.

## Integrated candidate

`candidate/parking_scheduler.py` shares the parking policy between the original
synchronous scheduler and upstream `AsyncScheduler`, preserving upstream token
accounting. Growth reservations include output placeholders and retain a high-water mark
when speculative rejection/draining shrinks the live estimate. Admission bounds
active plus retained terminal generations.

`terminal_state.py`, `terminal_versions.py`, and `terminal_worker.py` implement
selective terminal preservation: a device marker identifies eligible scheduler
EOS/stop-token/length endpoints; only the first terminal step copies recurrent
and circular state into independent packed buffers. Capture runs after target
postprocessing and MTP drafting. Generation-qualified ownership separates slot
retirement from payload release.

The generated completion, connector, GPU-copy and model-runner modules carry
terminal generations, retain them through allocation deferral, and select the
frozen payload for resident and disk-fallback capture. Circular destinations
are private; they cannot borrow a mutable live ring. The candidate launch script
adds an explicit async selector and deducts the snapshot reservation from the
existing KV-memory budget. TerminalWorker preallocates bounded payload storage
at KV initialization; retired slots cannot recycle it until generation release.
Lifecycle metadata uses pinned CPU staging and nonblocking stream-ordered
copies. Actual serving
launch/configuration are unchanged by this candidate.

## Checks completed

- CPU stopping oracle: 20,000 cases against deployed stopping rules.
- Actual Triton marker: 960 interpreter cases; immutable first-terminal selection.
- Actual copy kernel: normalization, temporal/circular state, padding, source
  overwrite and invalid-boundary cases. Both kernels compile for sm_121 offline.
- Ownership: bounded capacity, slot reuse, pending transfers, stale/duplicate
  acknowledgements and generation isolation.
- Integrated imports: sync/async method resolution and legacy positional metadata.
- Worker integration with real CPU tensors and Triton interpreter: descriptor
  binding, deferred snapshot surviving live-source overwrite and slot reuse,
  resident/disk selection and old-generation release. Toy geometry only.
- Completion allocation integration: actual BlockPool, DiskSlotManager and
  NativePressureCache; terminal completion with four tokens still in flight;
  repeated allocation deferral; both-rank preservation acknowledgement; private
  ring destination; retained generation/provenance; no premature disk fallback;
  exactly one finished notification. `async-completion-check.log` passed.

- Actual async scheduler/output methods with a real KV block manager and parking
  policy: two outstanding batches, quiescing one request while peers advance,
  draining placeholders, ignoring output after terminal stop, and three-draft
  rejection while a later batch remains outstanding. Transport is a facade;
  this does not test full checkpoint I/O or the hybrid allocator.
- Bounded worker pool: storage reuse after release, duplicate-release isolation,
  and retaining another generation's separate storage.
- `candidate.patch` applies cleanly to its recorded baseline and reproduces all
  nine replacement files. `PATCH_MANIFEST.json` records hashes and unchanged
  runtime overrides to preserve. Package is reviewable, not deployment-ready.

- Full candidate constructor with a real hybrid allocator and connector:
  terminal completion while another batch is in flight; generation retained
  through both completion acknowledgements; quiesce/drain pressure capture;
  same-pass parking and restore-load creation; restore promotion and new
  terminal generation. Worker outputs/ACKs are simulated, not transport I/O.
- Actual CUDA marker/copy and worker lifetime tests passed on rank 0 in a
  separate process with isolated buffers. A second run covers multi-tile
  copies (49,152 bytes per piece). Peak tensor allocation was under 2 MiB.
- A speculative-drain regression reproduced `ParkingPolicy.grow` rejecting a
  shrinking token estimate. The candidate now retains its reservation maximum;
  the regression passes. See `async-reservation-reproducer.log` and the updated
  `async-scheduler-check.log`.

- Hybrid cancellation scenarios passed for pending pressure saves and restores:
  ownership persists until acknowledgements drain; cancellation never resumes
  a request or leaks a terminal generation.
- All-deferred hybrid scheduling returns connector-only batches, retains bounded
  generation reservations across retries, and admits the same generations when
  allocation succeeds.
- Both-node launcher dry-runs validate both selectors, verified candidate mounts,
  and the 40 GiB synchronous / 38 GiB async KV-memory amounts without deployment.
- CPU and multi-tile CUDA worker scenarios also pass with pinned asynchronous
  lifecycle metadata staging (`device-state-async-staging.log`).

- The deployed Mamba and QSA binders accept reconstructed model shapes with
  the actual padded block stride on meta storage. TerminalWorker enumerates
  86 pieces and 59,109,824 content bytes/request, matching the estimate. This
  validates source-derived views without loading weights; it is not a live
  worker tensor inventory. Evidence: `model-layout-check.log`.

These checks are not full-model correctness validation or evidence of a
throughput improvement. The nonterminal fast-path GPU microbenchmark alongside
serving measured median 24.08 us at 12 requests and 51.78 us at 32 (p95 40.93 /
87.58 us), with 86 pieces per request and 30 samples. Contention and launch
spacing are included; this is added candidate work, not a causal throughput
comparison. Raw timings: `candidate-overhead.json`.

## Remaining deployment gates

1. Confirm the now-passing hybrid lifecycle, cancellation and all-deferred
   scenarios with actual model/worker execution, not simulated ACKs.
2. Validate descriptors against actual model tensor layouts and determine the
   packed byte footprint, including retained generations and metadata overhead.
3. Validate the reservation on the actual model/device. The source-derived
   estimate is 59,109,824 bytes/request, 1.762 GiB for 32 snapshots per rank.
   Proposed 2 GiB reservation reduces the 40 GiB KV pool to 38 GiB. The launcher
   deduction and bounded startup payload pool are implemented; allocator and
   full-model startup/device checks remain. See `terminal-memory-estimate.json`
   in the results directory; it is an estimate, not a live tensor inventory.
4. Full-model cached continuation validation; isolated device-copy checks pass.
5. Validate the assembled patch on the full model while preserving the deployed
   same-pass restore fix. Deployment remains conditional on validation and a
   promising performance result.

## Live service and observation

19:02 UTC: two additional paired hardware captures during natural drain have
completed; a corrected CUDA API wait probe now gives direct host-wait intervals
without CUPTI injection. Main-thread stream waits of 2.859/2.327 seconds overlap
only about 1.3/2.3 ms of sampled zero-SM time. A 3.608-second zero-SM span on both
ranks corresponds to the last request finishing and no queued requests until
the next client submission. Other short transition spans remain unattributed.
See MEASUREMENT.md for evidence and the failed-probe attempts excluded from
analysis. No observers remain running. Candidate and baseline hashes were
rechecked against all nine PATCH_MANIFEST replacements and match; serving is
still synchronous and healthy. No causal async speedup has been demonstrated.

18:52 UTC update: service health is HTTP 200 and the finite campaign is draining
(owner reports 20 complete, four running). No async deployment or serving tracer
injection has occurred. The disposable timed CUPTI prototype that finalized from
a driver API exit caused a subsequent CUDA launch failure, despite apparently
successful finalize; do not use it on serving. Its source is preserved as
`timed_trace.finalize-failed.cpp`. The revised prototype disables collection
without finalizing, but has not passed: both its first run and recheck failed
before collector initialization. A driver-only check isolates this to
`cuDevicePrimaryCtxRetain` returning CUDA_ERROR_OUT_OF_MEMORY (2), before any
test tensor allocation. Existing serving contexts continue generating. No orphan
test CUDA process was shown by nvidia-smi. This is a limitation of additional
context creation at present, not evidence of an async candidate failure.

`timed-cupti-disable-recheck.log` records the repeat failure. Do not inject the
unvalidated revised collector. Admission telemetry analysis now supplements
the sampled-idle measurements; see MEASUREMENT.md. Strict waiting-order and
restore dependencies mean async scheduling alone is not a remedy for every
observed admission delay.

The original synchronous engine crashed independently at 17:35:20 UTC during
same-pass park/resume. The narrow baseline fix is installed on both ranks.
The recorded readiness check returned HTTP 200 at 17:48:08 UTC. Live logs subsequently show request
`chatcmpl-b4d814fc6809dcdc-9310d0c0` parked at 17:57:30 and restored at 17:57:33,
boundary 46078, replay one token. This confirms one live recovery cycle, not
exhaustive parking validation.

The lower-concurrency observer (exec session 98768) exited normally at
18:11 UTC after its 30-minute deadline without two 8–14-request observations.
Its final observation was 24 running requests. No low-concurrency capture was
made in that window. The workload owner reports all 24 trials still running,
recent concurrency 20–29, and no reliable drain ETA. The replacement observer (session 13053) completed successfully. It captured
both ranks at 18:34 UTC while requests ranged 12–16. Host handoffs averaged
6.0/6.7 ms, but zero-SM samples within them represented only about 9.4/9.1 ms
per rank across the 30-second capture. See MEASUREMENT.md. No observer remains
running. GPU activity still cannot distinguish useful work from communication
waiting. The disposable-process late-CUPTI prototype now captures 21/21
expected timed kernels, including pre-existing CUDA graph replays, with no
dropped records. Graph execution remains valid after detach. This prototype
requires explicit CUDA synchronization before stop and is **not safe for live
injection yet**. Next: callback-exit detachment and disposable active-process
attachment tests. It has not been injected into serving. Source:
`live_trace_probe.cpp`, `check_late_cupti.py`; evidence:
`late-cupti-check-v4.log`, `late-cupti-v4.jsonl`. Earlier prototype reports only
validated API returns and produced no records; v4 adds strict kernel-count
validation and fixes the initially uninitialized record iterator.

Earlier high-concurrency measurements show largely overlapped host handoffs;
low-concurrency activity has now been measured; causal async benefit and
communication-wait attribution remain unresolved.
