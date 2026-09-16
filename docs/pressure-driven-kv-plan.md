# Pressure-driven KV disk paging

Status: implementation plan, 2026-09-13. No runtime changes applied.

## Requested behavior

Keep KV state in memory while the scheduler can accommodate the workload.
Write to disk only when a request must relinquish its resident memory. Current
requests will finish naturally: do not pause admissions, stop load generators,
cancel requests, or restart serving before they finish.

The deployment gate is both a validated replacement and a naturally idle server.
An idle server alone is not a reason to deploy an unvalidated change.

## Evidence and current coupling

- Live deployment: `/home/jon/dual-spark-inference-kv-paging`, TP2 across
  spark-1 and spark-2, 40 GiB resident KV and 48 GiB disk budget per rank.
- Both local PLE files and disk KV live on `/dev/nvme0n1p2`.
- The inherited offloading scheduler builds stores for newly eligible chunks
  during ordinary scheduling, without a memory-pressure condition.
- `offload_prompt_only=false` includes generated tokens. Completion snapshots
  additionally save state at request completion.
- A ten-second sample showed 28.27% resident occupancy, no preemptions, and
  connector-reported rates of 125.04 MiB/s saved and 64.68 MiB/s loaded. These
  are logical transfer counters, not measurements of physical NVMe throughput
  or proof of a particular decode slowdown.
- `GB10ParkingScheduler._try_park` currently pins an already saved aligned
  checkpoint. The scheduler waits at most 30 seconds for one. Its nominal
  SAVING acknowledgements are issued only because all-worker transfer
  completion has already happened in the connector.
- Old recurrent and windowed states are discarded as decoding progresses.
  Suppressing stores and then trying to recreate an arbitrary historical
  checkpoint is unsafe.
- The source tree contains a pause endpoint, but the running API does not
  advertise it. No pause endpoint is needed for the user's natural-drain plan.

## Implementation design

### 1. Make disk persistence an explicit policy

Introduce a configuration value such as `disk_write_policy`, with validated
values `pressure` and `eager`. Select `pressure` for this deployment and retain
`eager` for rollback and comparison. Reject unknown values.

In pressure mode, suppress ordinary incremental stores, aligned-boundary
stores, partial-tail stores, and request-completion disk snapshots. Apply the
policy before storage allocation, job registration, staging copies, or worker
capture. Dropping already-created jobs would corrupt lifecycle accounting.
Continue processing existing loads, acknowledgements, fences and cleanup.

Keep ordinary resident prefix caching. Completed conversations will no longer
receive guaranteed exact disk snapshots for follow-up reuse. Any later effort
to retain equivalent exact snapshots in memory must charge their full footprint
to the memory budget; it is not a hidden addition to this change. Previously
parked disk snapshots may still be read when needed, even at low occupancy.

### 2. Trigger saves from an actual reservation failure

Use the existing physical-block admission/growth ledger rather than a fixed
occupancy percentage. A failed growth reservation starts victim selection;
ordinary waiting work can remain queued. Recheck growth before committing a
save, since a peer may have finished and released its reservation.

Choose a resident request whose release enables progress, while retaining the
existing resume-growth allowance and avoiding repeated park/restore cycles.
Serialize saves initially to bound staging memory and disk contention. Keep
other requests progressing where their reservations permit it.

### 3. Capture a paused request from its current resident state

Replace the dependency on a pre-existing disk checkpoint with an explicit
quiesce -> capture -> save -> commit -> release lifecycle. Reconcile all
speculative outputs before capture and bind the snapshot to the accepted token
sequence and a request generation.

The preferred implementation extends the existing completion snapshot machinery
to capture an active, quiesced request. It already handles full attention,
windowed attention, recurrent state, and compression rings. This is a design
direction, not a claim that completion snapshots can be reused unchanged:

- Split snapshot creation from request-finished publication and cleanup.
- Capture current state at both aligned and unaligned token positions.
- Prove the correct accepted MTP state and recurrent convolution slice can be
  selected after outputs drain, including speculative rejection and boundaries.
- Preserve every GPU source and the worker request slot until both ranks have
  acknowledged the save; partial success must never release resident state.
- Keep snapshot keys pinned until restoration or cancellation has drained.
- Give active-request snapshots their own completion path: never emit a
  request-finished notification merely because a parking save completed.

If exact current-state capture cannot be validated, do not fall back to silently
enabling eager writes. An alternative is a bounded in-memory retained checkpoint,
but its state coverage, memory charge and replay bound need explicit validation.

### 4. Restore, cancellation and failures

Restore from the request-bound snapshot descriptor rather than generic prefix
lookup. Allocate private mutable state, reconstruct the required group layout,
wait for both ranks, and resume from the snapshot's committed position. Retain
the accepted output token sequence; do not duplicate streamed output.

Replace the old wait-for-an-existing-checkpoint timeout with actual transfer
progress tracking and a bounded I/O failure policy. Cancellation during capture,
save or restore must drain ownership before freeing blocks or disk pins. A
failed save must not free the only valid resident copy; a failed restore must
not promote the request. Preserve the current explicit error behavior rather
than introducing silent recomputation.

### 5. Diagnostics

Expose why a pressure save started, selected request generation, bytes queued,
save/restore progress, resident blocks released and failure counts. Distinguish
pressure saves from legacy prefix/completion writes. Avoid token content in logs.
Zero new disk-store bytes during a low-pressure workload is a release criterion.

## Code and packaging

Primary source modules are the GB10 parking scheduler, aligned connector,
completion snapshot helper, parking policy and rank-local disk transport.
Inspect the inherited offloading scheduler's store hooks before choosing where
to install the policy; patch the source and bundle explicitly if required.

Develop in an isolated checkout so the live bind-mounted overlays are not
modified. Preserve the existing uncommitted launcher changes and experiment
overrides. The current source checkout reports revision `01b52a02e5`; verify
the actual runtime files and both node configurations before building, since
image code and overlaid files need not match that checkout in every module.

Extend the existing parking policy, parking scheduler, aligned connector,
completion and disk transport suites. Build a separate deployment bundle using
the committed-source/hash manifest workflow. Save the exact current configuration,
overlays, manifest and service targets for rollback, including local launcher
changes. Do not regenerate the live bundle during development.

## Validation gates

1. Policy/lifecycle tests: low-pressure prefill, decode and completion create no
   stores; loads and cleanup still progress; pressure creates a real save;
   memory stays reserved until both acknowledgements; stale acknowledgements,
   cancellation and disk failures cannot free live state.
2. Snapshot tests: aligned/unaligned boundaries, MTP acceptance/rejection,
   recurrent state, rings, windowed pages, and long-context state coverage.
3. After natural drain, use both GPUs for a candidate canary. Force pressure
   with a reduced test reservation budget and prove actual save/release/restore
   and resumed output continuity. Exercise failure and cancellation paths.
4. Compare resumed output/log probabilities with uninterrupted controls at
   deterministic settings; account for the existing numerical variability.
5. Run a low-pressure workload through prefill, decode and completion and require
   zero new store bytes. Compare decode throughput/latency and physical device
   reads/writes using matched workload, concurrency, context and sampling.
6. Check health, aliases, streaming, cache reporting, long-context capacity and
   both node manifests before reopening normal service after the canary.

GPU tests must wait for the current requests to finish; development and CPU
validation can proceed while they run. If the candidate fails, restore the exact
outgoing bundle and report the failure rather than leaving a partial deployment.

## Natural-drain rollout

Do not change the current workload or admissions. At planning time there were
10 running requests and zero waiting; that is an observation, not a drain signal.
Monitor running and waiting counts, then verify parked/restoring requests,
pending connector work and HTTP response completion are also drained. Prevent
new admissions only at the final idle cutover if needed to avoid a check/restart
race; never abort an accepted request. If more requests arrive before cutover,
wait for them too.

Once the candidate is ready and natural drain is confirmed, stop the head service
using the existing coordinated worker shutdown, stage matching bundles on both
nodes, run the canary, and start the accepted configuration. A service restart
invalidates the existing same-process disk index; it cannot preserve active work.
No application or restart has occurred as part of this planning step.
