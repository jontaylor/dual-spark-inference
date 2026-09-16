# Async parking support: source audit and required changes

This is the source-level implementation specification. A mechanically applicable
`candidate.patch` and `PATCH_MANIFEST.json` now exist, but full-model validation
and deployment gates remain open. Serving remains synchronous. Do not deploy by removing the
two guards. Current files referenced below are frozen runtime inputs; make new
candidate copies and update deployment overrides/hashes only after validation.

The selective terminal-preservation candidate and component checks now exist;
see [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for their limits and
the outstanding integration work. A separate synchronous park/resume crash
was fixed during this work; it was not caused by an async deployment.

## Scheduler and launch changes

1. Factor `GB10ParkingScheduler` into a cooperative parking mixin plus synchronous
   and `AsyncScheduler` subclasses. Preserve both modes for measurement and
   rollback. The async class must inherit upstream `_update_after_schedule` and
   `_update_request_with_output`, including speculative placeholder rejection,
   processed-token cache publication and decode-step eligibility. Support TP2,
   PP1, V2, FCFS and the existing MTP3 configuration first. Assert the tested
   queue-depth limit rather than silently allowing arbitrary lookahead.
2. In `parking_scheduler.py:schedule`, include output placeholders in the growth
   reservation: `num_tokens + num_output_placeholders + num_lookahead_tokens + 1`.
   Recheck the recurrent fixed-cost budget against multiple in-flight steps.
   Keep quiescing requests unscheduled until `num_in_flight_tokens == 0`;
   parking generation and both-rank save acknowledgements must still precede
   preemption and source release. Existing pressure saves must not resume a
   request just because a provisional reservation becomes available.
3. Add an explicit config selector in `launch_rank.py` for scheduler class and
   `--async-scheduling` / `--no-async-scheduling`. Keep the current synchronous
   default until the whole candidate passes. Both ranks must agree. Upstream
   engine batch queues and worker output threads already implement the async
   plumbing; do not replace those with a separate scheduling implementation.

## Completion state ownership — the substantive change

The relevant current sources are `completion.py`, `connector.py`,
`gpu_completion_copy.py`, `gpu_checkpoint.py`, `model_runner.deep.py`, the
scheduler output metadata and the request/worker state lifecycle.

The required identity is `(request_id, request_generation, execution_step)`.
An output from step N must select the checkpoint state belonging to N even if
N+1 has run. A mutable request-slot index or the latest GPU accepted-count
array is not an adequate identity.

Required operations and ordering:

- Before a later forward overwrites N's recurrent and circular-buffer state,
  preserve the eligible accepted endpoint for N and its GPU-computed boundary,
  acceptance count, state-column selection and block-table version. Keep the
  original truncated-stop eligibility rules; never advertise an unavailable
  endpoint as a cache hit. Preserve attention source-page lifetimes as well.
- Carry the immutable version identity through scheduler output, worker result
  and completion-save metadata. Use it in `capture_gpu_completions` instead of
  indexing the latest `runner.req_states` and `model_state` arrays.
- At a terminal output, select and retain that version **before** releasing
  request slots or notifying the connector that its sources can be freed.
  Later in-flight outputs drain without appending tokens to a finished request
  or changing the selected terminal boundary.
- Completion allocation deferral must retain this version and its restoration
  provenance, not only the old mutable request block IDs. Retry once capacity
  becomes available; preserve the current bounded timeout and error handling.
- Explicit release acknowledgements retire unused step versions. Retained
  completion versions transfer to the existing resident-cache ownership model.
  Ring-slot reuse, request-ID reuse and cancellation must not release a newer
  owner on a delayed acknowledgement from an older generation.
- GPU source copies and disk persistence remain separate lifetimes. Preserve
  the current rank-identified source-preserved acknowledgements and the
  all-rank disk publication gate. Do not reintroduce synchronous disk fences.

Only then replace the synchronous-completion rejection in
`connector.py:__init__` with a checked versioned-state capability contract and
make `CompletionCache.finish` accept an owned terminal version when later work
is still in flight. Simply deleting the in-flight guard loses correctness;
keeping it silently loses completion-cache functionality.

### Cost must be resolved before choosing the version-storage implementation

A naive two-slot copy ring for every request is not free. The observed physical
page size is 27,156,480 bytes, with four recurrent groups and one circular-buffer
group. If each retained version needs one physical page from each group, at
s32 a two-version reservation alone is 8,690,073,600 bytes (about 8.09 GiB).
Copying all five groups for every request every step would copy about 4.35 GB
per step before accounting for additional speculative endpoints. This is a
capacity/bandwidth estimate, not an implemented layout or measured copy cost.

A viable patch should use versioned source ownership/copy-on-write or selective
terminal-state preservation, and prove coverage of EOS, length stops, stop
strings and cancellation. GPU-only EOS detection alone does not cover arbitrary
CPU stop-string handling. Any reduced functionality must be explicit; disabling
completion reuse is not matching support. Compare the cost to the measured
overlapped host intervals, not a fictitious 5–6 ms completely idle GPU interval.

## Validation needed for an applicable patch

- Real scheduler tests with two outstanding batches: MTP acceptance/rejection,
  placeholder retirement, growth crossing 7,680 tokens, mixed prefill/decode,
  both-rank restore, all-deferred connector progress, quiesce/drain/park/resume.
- Terminal output N with N+1 already scheduled and already executed; exact
  retained endpoint and snapshot contents must match the synchronous eligible
  endpoint. Include EOS within speculative output and length/stop truncation.
- Deferred allocation after termination, slot reuse, cancellation during
  pending copies, duplicate/skewed/late rank acknowledgements and source reuse.
- Byte-level device state capture/restore tests, followed by full-model cached
  continuation checks. Existing numerical discrepancies must be distinguished
  from new regressions, not used to waive these checks.
- Bounded matched measurement of queued execution, GPU activity, accepted
  output tokens and latency at multiple concurrency levels. Validate startup
  really selects the async scheduler and worker output thread on both ranks.

Deployment is conditional on a promising result and a validated patch. Neither
an executable full-parity patch nor that deployment gate is complete yet.
