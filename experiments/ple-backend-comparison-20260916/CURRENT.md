Completed campaign checkpoint, 16 September 2026, 13:33 UTC.

SUPERSEDED: the user subsequently selected original external workers for
maintainability. Both controls were switched to external epoch 15, followed by
an external-only restoration which removes the comparison runtime entirely.
See ../ple-external-restore-20260916/README.md for current production state.
The remainder is the preserved campaign-completion checkpoint.

The original-worker versus in-process comparison is COMPLETE. Exec session
8830 exited successfully. No controller should be restarted against the same
result directory. Two warmups and eight measured cycles ran in one deployment.

Result: external 261.457 useful output tokens/s; in_process 262.574. Four paired
cycles estimate +0.4249% for in_process, preliminary 95% interval -0.7811% to
+1.6456%. No demonstrated throughput winner. In_process remains selected as
the incumbent; this is not a declaration of superiority. See FINAL_ANALYSIS.md.

Final validation: 120 requests including warmups, 108 per-lane repeats, zero
payload/output/token-count mismatches. Eight measured cycles had identical
24,576 output tokens and 42,302 uncached prompt tokens, independently verified
against server counters. 5,851 aligned completed steps with zero audit errors;
final sequence 5852 timing is buffered on both ranks until next request/close.
Client throughput includes all output. Both nodes passed frozen source/mount,
parameter/control/affinity checks. HTTP health 200; processes unchanged;
zero container restarts/OOM kills.

Current uint64 backend control on BOTH hosts:
/home/jon/.cache/vllm-ple-control/ple-backend-policy.bin
Value 60129574912: fixed in_process, mode 0, epoch 14, block_steps 128.
Independent uint32 ple-read-policy.bin is STILL ZERO: old SQPOLL, rejected
prefetch disabled. Do not confuse these files. Read/verify controls using
set_backend.py without --mode. Switch only while the client is held AND API
running/waiting counts are zero. No model restart is needed for a backend switch.

Serving sources backend_comparison.py, backend_control.py, in_process.py,
model_runner.py, gpu_worker.py and legacy_{connector,worker,protocol}.py are
LOADED FROZEN ASSETS: do not edit in place. Both producers share the original
connector's output allocation, mapped GPU pointer, completion flag and consumer
fence; only one executes each batch. Both caches remain resident. The native
library stays experiments/ple-resident-20260916/libple_batch_reader.so (SHA256
522a29d866c7f9c93bb86a4e468774220736503ec0b229cb34887a813420d761).

Settings s32/b8192/t1024, TP2/MTP3; all CPU affinities 0-19 unrestricted.
Head service qwen38-next-qwen-fp8.service, worker service on 192.168.100.11.
Head container PID413259, EngineCore413460, GPU worker413557, external worker
413813. Full both-rank process evidence is live-verification.json. Both external
worker processes remain available but receive no batches in fixed native mode.

Common per-step JSONL telemetry remains active at results/
ple-backend-comparison-20260916/live/rankN on each host. The former native-only
ple-serving-monitor is deliberately STOPPED because it misattributes backend
switches. Do not reenable it for cross-backend analysis. Completed evidence:
comparison.json, final-summary.json, repeat-identity-audit.json, final-health.json,
final-backend.json, live-verification.json, client.json and per-cycle .prom files.
Offline analyze.py and review.py reproduce the result. collect.py should run
only drained; it replaces collected snapshots. Future campaigns need separate
labels/results and a frozen plan; do not simply rerun campaign.py.

Remote client: jon@192.168.0.167, controller PID1557868, HELD_IDLE after ten
cycles, zero active requests. Directory:
/home/jon/artifact-agent-orchestration/.artifact-agent/ple-comparison/20260916-fixed-decode
Coordination status /tmp/ple-repeating-load-status.json. User-authorized remote
thread Review latest planner experiments (3), ID
01a09b9f-c6c7-7bb0-8c25-b22748f43388. The thread was informed of final results
and instructed to leave the client held. Use codex queue with shlex.join for
messages. No subagents, commits or pushes requested.

No further deployment work pending. Full rollback configs deploy.before-r0.json
and deploy.before-r1.json are retained. Install BOTH host configs before head
start because its ExecStartPre starts the remote worker. Do not daemon-reload:
unrelated unit changes are pending. Do not run standalone GPU tests alongside
serving. Both-backend validation previously passed 240 real exact GPU-byte tests.

Scope: synthetic warm c12, both caches resident, common instrumentation overhead
not quantified, output repeatability is not broad quality certification. Direct
decode readiness waits were tiny for both backends; startup/synchronization,
GPU embedding consumption, cold I/O and mixed-load contention remain separate
questions. Further work should follow exposed whole-step cost, not assume
removing IPC alone improves serving throughput.
