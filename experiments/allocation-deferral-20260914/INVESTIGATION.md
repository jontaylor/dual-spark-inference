# Allocation-aware deferral investigation — 2026-09-14

## Conclusion and scope

Feasible, with a concrete integration path. Not a configuration toggle or a change confined to the allocator. It requires coordinated changes to allocation planning, request scheduling, GPU-source ownership, and worker-to-scheduler acknowledgements. No serving code/configuration was changed and no restart was performed. No deferral implementation or performance test has been run. The asynchronous eight-buffer candidate remains the subject of this investigation.

Read installed scheduler, KV cache manager/coordinator, per-type managers, engine and connector sources into this directory. `live-overrides.json` records actual head-container Python mount paths and SHA256 hashes (27 Python mounts, including base patches and entrypoint; this is not a claim of 27 runtime overrides). The scheduler snapshot is the deployed custom scheduler. Current V2 runner source is `../targeted-determinism-20260913/model_runner.deep.py`; the captured legacy runner is supplementary, not the active runner. Relevant override paths match the async-eviction candidate.

## Verified current paths

1. `../gpu-native-completion-20260914/block_pool.gpu.py:647`, `get_new_blocks`: checks total free count, removes N blocks from the queue, invokes `before_cached_block_reuse`, evicts cached hashes, increments references. This is too late to reject an allocation cleanly: the queue is already mutated, and hybrid allocation can call it multiple times for different groups.
2. Same file `free_blocks:728`: truly uncached blocks are prepended; cached blocks and native completion pages are appended. Thus the current allocator already favours empty storage. It does not explicitly choose all no-transfer candidates ahead of all GPU-only completion pages. A search may help, but abundant alternative capacity is NOT established by current metrics.
3. `../gpu-native-completion-20260914/native_pressure.gpu.py:42`, `before_reuse`: creates native eviction jobs after allocation selection. `add_metadata` tags them for a flush before worker request-state changes.
4. `../async-eviction-20260914/rank_local_disk.async.py:209`, `begin_native_reuse`: asserts physical block refcount zero, takes a storage-key pin, marks the slot spilling, but does not reserve the physical GPU block. `lookup` returns HIT_PENDING while spilling. `finish_spill` currently releases resident metadata only after persistence ACK.
5. `../async-eviction-20260914/connector.async.py:493`, `prepare_completion`: calls `handle_preemptions`; current `jobs_to_flush` route submits and waits. The V2 runner invokes this before finish/free/add/update request state (`model_runner.deep.py:1571`), which can overwrite allocated blocks.
6. `core__kv_cache_manager.py:503–541`, `allocate_slots`: releases skipped blocks, calculates allocation demand, checks capacity, then adopts computed blocks and allocates new blocks. The boundary before `allocate_new_computed_blocks` is the useful admission/commit boundary. It is not entirely side-effect-free: skipped-block reclamation and Mamba sizing bookkeeping already occur.
7. `core__sched__scheduler.py:702`: a running request's `allocate_slots` returning None enters victim preemption. Our parking subclass rejects unplanned preemption. At :1144 a waiting allocation failure breaks waiting admission. Neither means "defer just this request and continue".
8. `../performance-until-10am-20260914/parking_scheduler.reservation.py`: holds quiescing requests out of the scheduled running list while keeping their state. This demonstrates per-request withholding but is a parking path, not transient allocation deferral. Its strict waiting order and `_quiescing` gate block new admissions. Do not reuse that flag for every preservation wait.
9. `core__sched__scheduler.py:2541`, `_free_request`: honours connector `delay_free_blocks`; `finished_sending` later frees the request (:2980). Completion save already uses this lifetime protocol.
10. `../gpu-native-completion-20260914/completion.gpu.v2.py:253–274`: tries memory checkpoint allocation, then falls back to disk on None. A deferred memory allocation cannot masquerade as ordinary None without initiating that fallback.
11. `distributed__kv_transfer__kv_connector__v1__offloading__common.py:76`: worker metadata carries final completed-job counts aggregated across ranks; it has no source-preserved acknowledgement. The worker-local preserved future is insufficient to release scheduler-owned GPU blocks.
12. Active V2 runner `model_runner.deep.py:1579` calls connector no-forward handling when zero tokens are scheduled. `core__sched__scheduler.py:2636–2663` keeps delayed-finish requests alive, and engine `step` executes scheduling while requests remain. A deferral design can use this infrastructure but must explicitly ensure outstanding preservation work keeps it alive after cancellation/last-request retirement.

## Proposed scheduling sequence

Use a distinct `WAIT_FOR_PRESERVATION` result, separate from physical OOM/capacity exhaustion.

1. Determine the request's required allocation across all groups before committing any group. Separate existing prefix blocks that will be adopted from fresh destination blocks. The current demand count includes some refcount-zero cache hits removed by touch, so treating that number as entirely fresh allocations would overestimate demand.
2. Identify eligible blocks that require no external preservation, excluding blocks the same request will adopt. Prefer uncached blocks, then use an explicit policy for other cache eviction; do not silently sacrifice valuable local prefixes merely because they require no disk copy.
3. If sufficient eligible blocks exist, reserve the selected set for this allocation and commit atomically across groups. No preservation transfer is introduced on this fast path.
4. Otherwise reserve only a bounded set of eligible idle GPU-only checkpoint pages for preservation. Remove them from allocatable capacity and hold an explicit physical source reference. Do not assign them to a destination request yet. Keep actual capacity shortage distinct from temporary withheld capacity.
5. Submit source-copy work without `jobs_to_flush` blocking the worker. Record job ID, source block/generation and all relevant storage slots. A single block may back multiple slots; releasing it requires preservation of every dependency.
6. Omit the affected request from this batch without freeing its state, altering accepted token progress, or preempting a victim. Continue considering other running requests. For waiting requests, use a bounded skipped/retry mechanism; the existing strict head-of-line policy must be handled explicitly. Retain priority/age so repeated newcomers cannot starve a large deferred allocation.
7. Report source-preserved ACK exactly once per job per rank, separately from persistence ACK. Aggregate both ranks before detaching old physical-source ownership and returning the block to eligible allocation. This acknowledgement may arrive in a later scheduling step; no same-step overwrite is necessary.
8. Retain the backing slot as pending until final persistence completion. Loads must still wait. Disk failure still propagates through final completion; early source ACK is not a durable-cache hit.
9. Retry deferred allocations with fair access to newly released capacity. Run connector-only steps to collect ACKs when no model request can progress. Avoid a hot polling loop.

Example: A needs one fresh block but only a preservable block is available; B can decode using its existing blocks. Reserve and copy the old page, defer A, run B. Once both ranks acknowledge source preservation, A can receive the block while persistence continues. If B also needs unavailable capacity, both must wait for copies; that is genuine capacity dependence, not an unconditional worker fence.

## Completion allocation is separate

A finished request can retain source blocks using the existing delayed-free mechanism while awaiting snapshot allocation. Add an explicit pending-allocation state before the existing pending-transfer state. On temporary shortage, retain request state and provenance and retry later; do not interpret it as disk fallback or discard.

`finish` currently pops selected/restored provenance, allocates unique keys and pins shared pages. Re-entering it from the beginning without preserving or rolling back that state is unsafe. Build/store a stable plan or implement a clean retry boundary. Only publish after all snapshot dependencies complete; only emit finished_sending once.

Finished requests waiting for checkpoint allocation hold GPU capacity. A bounded policy is required: reclaim idle checkpoint victims first, bound pending snapshot allocations, and preserve the existing genuine-capacity fallback when there are no reclaimable victims. Otherwise many finished requests could hold all source blocks while each waits for snapshot destinations. Suspension checkpoints retain their stricter state-preservation requirements. Simply delaying every completion indefinitely is not a solution.

## Required ownership changes

- Do not only add a refcount in `get_new_blocks`: it has already popped destinations, and current assertions require zero refs before native reuse.
- Split resident source ownership from backing-slot persistence. Current `_release_resident` assumes spilling has ended; a source ACK must detach the GPU identity without advertising a ready disk hit. Final ACK must not later free a block already reassigned to a new request.
- Prevent late prefix lookups/COW from mutating a reserved source. Invalidate or withhold mutable cache aliases at reservation, or prove that every remaining hit is read-only until source ACK. Refcounts alone are not a complete alias policy.
- Key releases by job and allocation generation, handle cancellation, protect multiple slot references, tolerate ranks ACKing in different steps and do not count duplicate ACKs twice.
- Existing copy worker waits on captured CUDA readiness events. Retain that ordering for sources touched by previous model work; start copies on a suitable independent stream without inserting a blanket compute wait.
- Reservation/free metrics and parking ledger must reflect real ownership. Do not count reserved copy sources as immediately allocatable, and do not count borrowed GPU blocks twice.

## Why 64 buffers is not equivalent

A larger CPU staging region absorbs larger bursts, but the current worker still waits for all required source copies. Allocation-aware deferral allows unrelated requests to run while those copies wait for a buffer or finish. It does not remove DMA bandwidth costs, CPU hashing, disk traffic, true global capacity exhaustion or the need for bounded staging. Eight buffers can still throttle how fast deferred requests resume; increasing them is an independent measured tradeoff.

## Validation needed before deployment

No new serving tests were run because this turn is a source-level investigation, not an implementation.

1. Held-persistence/held-copy tests: A deferred, B advances; A obtains no source block early; release only after both rank source ACKs; disk lookup remains pending until final ACK.
2. No-transfer fast path: uncached alternatives, zero-extra-block decode, local cache adoption, Mamba partial-hit/checkpoint/COW allocations and MTP lookahead counts. No partial group allocation on a deferred result.
3. Exhaustion: all requests deferred, no eligible victims, staging full, only pending finished requests, cancellation of last live request. Check no deadlock, hot spin or unplanned preemption.
4. Ownership: multiple slots/block, prefix lookup during copy, rank skew, duplicate ACK, stale-generation ACK, copy/write failure; exact byte restore after source reuse.
5. Completion: repeated temporary failures preserve provenance and pins; no duplicate save or finished_sending; no unintended disk fallback; bounded finish backlog.
6. Live C1/C4/C10 token/logprob/cache-boundary comparisons, aged disk restore, and unchanged representative pressure workload. Existing cold/cache numerical difference remains a separate unresolved issue.
7. Record per-request deferred reasons/age, candidate counts, source-reserved blocks, source-copy queue/wait, CPU-buffer exhaustion, generation throughput, prompt reuse, disk bytes and GPU low episodes. Compare equivalent windows; current captures do not establish a 15 tok/s loss or a predicted gain from deferral.

## Unknowns

Number of transfer-triggering allocations with a usable alternative block; split between completion, decode and prefill triggers; how much source-copy waiting can overlap useful unrelated compute; effect on fairness/local hit rate; actual throughput benefit. Existing logs do not contain exact free-queue snapshots or allocation-attempt demand, so these cannot be inferred reliably from low GPU utilisation or reported KV usage.

Recommended implementation direction: explicit allocation preflight and transient deferral, two-stage rank ACKs and physical source reservations, plus the separate completion retry path. Preserve the current reference-count and final persistence semantics; do not remove the fence until all allocating paths use the new protocol. No automatic restart or deployment is justified by source inspection alone.
