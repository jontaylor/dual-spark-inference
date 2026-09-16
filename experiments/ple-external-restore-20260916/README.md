# Original external PLE workers restored — 16 September 2026

The user selected the original external-worker implementation for maintainability
after the complete warm c12 comparison found no demonstrated throughput winner
(external 261.46 versus in-process 262.57 output tokens/s; paired difference
+0.42%, preliminary 95% interval −0.78% to +1.65%). This restoration is an
operational choice, not a new performance claim.

Both nodes now run the **original external-only path**: one PLE process per node,
ZeroMQ batch notification, native CPU hashing, 16-thread mmap gather, a 1 GiB
row cache and the original mapped GPU output/completion transport. Connector,
worker and protocol source bytes match the original image and the external arm
of the comparison.

The inactive in-process reader, GPU hash library, dual-backend wrapper, policy
file mounts and custom io_uring seccomp profile are removed from the deployment.
The GPU runner and PLE layer overrides were restored to their verified pre-PLE-
experiment versions. Other runtime overrides and serving settings were retained:
**s32/b8192/t1024, TP2/MTP3**, BF16 KV, existing paging policies, and unrestricted
CPU affinity 0–19 on every thread. The SSD latency policy remains unchanged.

The restoration first selected external epoch 15 in the running comparison
deployment. Both external-only launch configurations then passed dry-run checks
and source-hash verification before installation. Both host configs were
installed before restarting the head service, whose ExecStartPre starts the
remote worker. Exactly one coordinated reload was needed. No systemd
daemon-reload was performed; unrelated pending unit changes remain untouched.

**Validation passed:** both original external workers are present; all mounted
source hashes and serving parameters match; native reader/hash libraries are
absent from process mappings; custom seccomp and comparison mounts/environment
variables are absent; all inspected threads have CPUs 0–19. API health returned
200, unauthenticated access returned 401, both aliases worked, four concurrent
arithmetic requests passed, a weather tool call passed, and streaming returned
READY with a completed stream. These are functional checks, not a throughput
retest or broad quality certification.

Published evidence: [live-verification.json](live-verification.json) and
[smoke.json](smoke.json). [deploy.example.json](deploy.example.json) is a portable
copy of the selected config, also installed as the repository's main example.
Adapt host paths, interfaces and image identities for another machine. The
published server submodule and paging manifest pin
`318e06e84d2aeb70a240531705ae67fa3f977b7a` on the user's
[vllm-gb10 fork](https://github.com/jontaylor/vllm-gb10/tree/gb10-pressure-paging).
All selected overlay sources are included in this repository and match the
example config's hashes. Local config backups and raw captures are not published.

The repeating comparison client remains held idle on jon@192.168.0.167. Its
owner thread was informed that serving is healthy. API/Prometheus and client
token timings remain available. The dual-backend per-step logger is no longer
installed, and the obsolete native-only PLE monitor remains stopped. Switching
backend-control files does not change this external-only runtime. A future
comparison requires explicitly deploying the archived comparison configuration.

Recheck the live deployment:

```
.venv/bin/python experiments/ple-external-restore-20260916/verify_live.py
.venv/bin/python smoke_api.py results/ple-external-restore-20260916/smoke.json
```

The second command sends a small functional load. Local rollback configs are
`deploy.before-r0.json` and `deploy.before-r1.json` beside this record. Restore
both matching host configs before head startup to return to the comparison
deployment; retain its frozen files and native libraries. Do not modify loaded
sources in place. The measured comparison and candidate implementations remain
archived, not enabled by the production configuration.
