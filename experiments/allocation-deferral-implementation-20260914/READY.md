> Activated after preparation: see [ACTIVATION.md](ACTIVATION.md) for the current running state and live results. The preparation record below describes the pre-activation checkpoint.

# Allocation-aware deferral candidate: ready for controlled live testing

Implemented and staged on both nodes, 14 September 2026. **Not activated.** The existing async-eviction server remains running and returned HTTP 200 on its health endpoint after preparation. No serving configuration, model, sampling, cache policy or service restart was changed during implementation.

## What is implemented

The allocation gate runs before multi-group allocation commits. It prefers blocks requiring no checkpoint preservation, while protecting the prefix-hit blocks the same request will adopt. Empty/zero-extra-allocation and already-eligible queue-head cases avoid a whole-cache scan. If required GPU-only checkpoint pages must be preserved, the gate reserves their physical blocks, removes old cache aliases, submits copy jobs, and raises a distinct transient allocation result. Running and waiting scheduler paths catch that result and continue considering other requests instead of victim preemption or unconditional batch exit.

Reserved jobs use a separate connector metadata field. They are submitted before worker request-state changes, **without calling handle_preemptions or wait**. Sources remain physically reserved until each required copy has an acknowledgement from both ranks. Source acknowledgements carry rank identities and aggregate separately from persistence completions. The last required source acknowledgement detaches the old GPU identity and releases the physical reservation; disk keys remain pending until persistence finishes. A late persistence ACK cannot free a newly assigned GPU owner.

Completed requests can defer snapshot allocation using the existing delayed-free lifetime. Their source blocks and restore provenance remain held; temporary shortage neither discards the checkpoint nor triggers disk fallback. Retried snapshot allocation creates one save, then one finished-sending notification. A slotted request-state field keeps the offloading scheduler's state alive during that wait. Genuine memory-capacity failures retain the existing disk fallback.

Source reservations are capped at 64 physical GPU blocks concurrently and are created only in response to allocation demand. **This is not a change to 64 CPU staging buffers.** The existing eight CPU buffers (217,251,840 bytes/rank) remain unchanged. Sources release after copying, not after persistence. When all requests need unavailable capacity, an empty model batch lets connector work/ACKs progress; the existing engine yields 1 ms on no-model steps.

Legacy non-native eviction fences remain for the legacy path; this candidate removes the unconditional native-completion eviction fence in the current native configuration. Source copies, disk loads, genuine capacity exhaustion and GPU bandwidth contention can still delay progress. No claim that every GPU dip disappears.

## Validation completed

Seven suites passed; machine-readable details are in `checks.json`, with individual logs:

| Suite | Evidence |
|---|---|
| Allocator/ownership | `ownership-check.log`: source reservations; unrelated zero-allocation progress; duplicate/late and skewed-rank source ACKs; disk publication gate; final ACK preserves new owner; prefix adoption; bounded multi-step preservation; multiple storage keys per physical page; source/public-future race; copy-error propagation; metadata aggregation |
| Actual scheduler, no model weights | `scheduler-check.log`: deferred waiting A while B scheduled; deferred running A while C scheduled; no victim preemption; all deferred returns empty model batch and keeps engine active |
| Actual completion planner | `completion-check.log`: repeated allocation deferral retains source/provenance; no premature transfer job; two-rank copy ACK permits memory retry while disk pending; one completion notification |
| Actual hybrid KV manager | `hybrid-check.log`: no group destinations allocated on deferral; complete attention/recurrent allocation after source ACK with three speculative state blocks; persistence still pending |
| Connector integration/import | `connector-check.log`: all candidate modules import together; real slotted state accepts deferral field; reserved submission invokes neither wait nor handle_preemptions; metadata drained once |
| Rank0 GPU | `gpu-r0.log`: real 27,156,480-byte page; unrelated GPU work completes with disk write held; source ACK before persistence; overwrite original then exact checked disk restore |
| Rank1 GPU | `gpu-r1.log`: same real-page test on worker GPU |

The scheduler suite uses a tiny local model *configuration* only; it does not load model weights. GPU tests are isolated diagnostic processes and have exited; their shared-memory regions were cleaned up. Ownership/hybrid tests simulate rank ACK arrival without loading the model; GPU tests validate real transport on each rank separately. These are not full-model live acceptance results.

## Candidate changes

Twelve candidate source files are mapped in `manifest.json`; unified before/after diffs are in `diffs/`. Runtime override entries increase from 23 to 28 on each rank (seven existing runtime overrides replaced; five entries added, including an existing base-patched offloading scheduler now explicitly overridden).

| Candidate file | Responsibility |
|---|---|
| `deferral.py` | Distinct transient allocation exception |
| `native_pressure.py` | Allocation gate, physical reservations, source ACK aggregation and release |
| `block_pool.py` | Guard direct allocations before queue removal |
| `kv_cache_manager.py` | Hybrid preflight, prefix-adoption protection and temporary capacity checks |
| `scheduler.py` | Continue past transiently blocked running/waiting requests |
| `rank_local_disk.py` | Detach preserved GPU identities, retain pending backing, poll source ACKs |
| `common.py` | Rank-identified source ACK metadata |
| `worker.py` | Report source and final ACKs without losing a same-step completion |
| `completion.py` | Pending-allocation retry and provenance retention |
| `connector.py` | Nonblocking reserved job submission and metadata propagation |
| `parking_scheduler.py` | Retry held completion allocations and clean deferred-request records |
| `offloading_scheduler.py` | Keep request state alive while snapshot allocation waits |

`before-r0.json` / `before-r1.json` preserve the serving configurations at preparation. `candidate-r0.json` / `candidate-r1.json` change only override mappings and their hashes. MTP3, numerical kernels, cache retention, model and batching settings are identical. `staging-verified.json` confirms all 28 configured source hashes on both nodes. Candidate sources are staged at identical absolute host paths.

## Activation and rollback

Prepared but NOT executed:

```bash
python3 experiments/allocation-deferral-implementation-20260914/activate.py
```

Activation rejects configuration drift and mismatched source hashes before stopping anything; it coordinates with the existing workload task, applies both rank configs and requests one service restart. It does not equate a requested restart with API readiness or live-test success. An activation write/start error invokes the prepared rollback script.

Exact prior-config restoration:

```bash
python3 experiments/allocation-deferral-implementation-20260914/rollback.py
```

Neither activation nor rollback ran in this implementation turn. Do not rebuild or edit these source files after activation; create a new version if changes are needed.

## Live acceptance still required

- Verify actual mounts/hashes and readiness after activation; keep the representative workload's cadence, sampling and retries unchanged.
- Run full-model C1/C4/C10 token/logprob/cache-boundary comparisons and aged disk-restore replay.
- Observe real pressure: source reservations/ACKs should occur while unrelated requests generate tokens; native eviction jobs must not enter the pre-forward flush wait.
- Compare aggregate generation, concurrency, prompt reuse, disk traffic and GPU dips over comparable windows. No speedup or quantified recovered token/s is claimed yet.
- Exercise sustained mixed prefill/decode pressure, completion backlog, cancellation while deferred and prolonged allocation fairness. Those end-to-end combinations are not fully covered by the component tests.
- The existing cold-versus-cached numerical discrepancy remains outside this transport/scheduler change; deterministic serving is not declared universally verified.
