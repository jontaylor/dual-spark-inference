# Allocation-aware deferral activation — 14 September 2026

User explicitly requested activation. Candidate activated and serving on both ranks. API ready at 18:58:53 UTC. All 15 live token/logprob/cache-boundary comparisons passed. Candidate left running.

- Activation preflight verified unchanged prior configs and all candidate source hashes before stopping serving.
- Existing representative workload task notified to retain sampling, cadence and retries.
- One service stop/start performed by `activate.py`; no daemon-reload or service-unit edits.
- Containers started at 18:51:13 UTC on both ranks.
- `runtime-verified.json`: all 28 runtime override mounts and in-container SHA256 hashes match both candidate configs.
- `live-r0.json` and `live-r1.json`: actual container arguments, environments, mounts and state, with API secret redacted.
- Actual arguments on both ranks: TP2, max-num-seqs 32, max-num-batched-tokens 8192; speculation method mtp, num_speculative_tokens 3, use_local_argmax_reduction false, draft_sample_method probabilistic.
- No change to CPU staging capacity: eight buffers/rank. Up to 64 GPU source blocks may be temporarily reserved in response to allocation demand; these are existing KV blocks, not new CPU staging buffers.

Pre-activation component/GPU checks are documented in READY.md and checks.json. They do not replace live model acceptance tests. Full-model C1/C4/C10 token/logprob/cache-boundary verification is queued after readiness. Performance benefit is not yet measured. Cold/cache numerical parity remains a separate unresolved issue.

Activation/readiness logs are preserved. The unit-changed warning during stop/start existed before this candidate; no unit reload was introduced as part of this activation.

## Live acceptance result

The correctness probe passed all 15 comparisons (C4, C10 and final serial) against the serial reference. The seed contained 7,264 prompt tokens; continuation 7,428; the expected 7,327 cached tokens were reported consistently. Generated token IDs and returned logprobs matched exactly. See the latest `correctness-*/summary.json` and `live-correctness.log`.

Disk store and load deltas during these checks were both zero. No native reservation/deferral event was observed during this initial acceptance window. This verifies continued serving and concurrency parity, not live eviction-pressure performance. Background representative load resumption was not confirmed during the probe; the existing workload task was notified that the API was ready and asked to confirm resumption without changing settings. No aggregate-throughput improvement is claimed.

Final startup/runtime scan found no traceback, assertion error, CUDA OOM or ERROR log entry on either rank. The candidate remains active; rollback was not used. Full disk-pressure and sustained workload validation remains pending.
