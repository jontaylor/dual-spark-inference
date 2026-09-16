# In-process PLE batch lookup

The opt-in `optimizations.ple_in_process` mode executes PLE hashing, read submission, completion handling and row gathering inside each GPU worker. It skips creation of `PleOffloadWorker` and does not create a PLE ZeroMQ context, registration socket, or request socket.

The lightweight GPU placeholder retains its constructor metadata and exact checkpoint hash buffers. Existing FP8 scale loading/dequantization remains in the GPU model. No second model or anonymous copy of the large table is constructed.

The native reader opens the local packed file, deduplicates rows, checks its bounded exact-row cache, and submits every missing row through `io_uring` with `IOSQE_ASYNC` before reaping/waiting for completions. A ring pool is sized at startup for the configured maximum token batch (16 rows/token here). Cache hits do not require I/O. The current implementation also uses this reader for prefills; it does not silently fall back to serial mmap faults or OpenMP. The kernel controls physical I/O concurrency; in-process refers to application ownership, not elimination of kernel I/O workers.

Completed unique rows are held in native scratch, then scattered in original order into the existing mapped GPU-accessible output. This introduces a small scratch-to-output copy, but allows a failed or short read to leave output uncommitted. This is not a claim that ready data is already resident in GPU L2. GPU consumption still follows the release/acquire completion flag. The preceding consumer event protects output reuse.

Limits: the new mode deliberately validates the current node-local TP2/DP1, single-PLE-layer, packed-FP8 configuration. It does not silently accept unsupported model topologies or row formats. Native cache is bounded to the configured 0–1024 MiB budget. No global offload-process marker is changed.

On I/O error or timeout, the step fails and never publishes ready. Handles with pending reads retain their native scratch memory; destruction returns EBUSY rather than freeing live targets. Python retains such handles until process exit if outstanding completions cannot be reaped. Retrying a poisoned reader is rejected. This favors correctness over recovering a failed serving process in place.

## Container policy

`io_uring_setup` returned EPERM under the existing container profile. `seccomp-uring.json` is the preserved Moby default-profile snapshot in `seccomp-base.json`, with only io_uring_setup/io_uring_enter/io_uring_register rules replaced by unconditional allows. Source fetched from https://raw.githubusercontent.com/moby/profiles/main/seccomp/default.json on 2026-09-16; both snapshots are retained. The launcher verifies SHA-256 of the policy and native library. A separate container with this policy successfully opened an io_uring. This does not disable seccomp globally.

## Checks completed before rollout

- Native byte differential checks: repeated IDs, page-straddling 160-byte rows, 9,000 distinct rows across three queues, warm exact cache and no row cache, invalid bounds and short-file reads. Three pytest cases passed.
- A cold-file syscall trace submitted 4096 + 4096 + 808 reads without any blocking completion-wait syscall between submissions. Evidence: `../../results/ple-batch-implementation-20260916/submit-before-wait.strace`.
- Separate GPU container exercised CPU/CUDA input staging, actual mapped GPU reads, repeated buffer reuse, checkpoint hash overrides, and dummy output handling. Both staging modes passed; `integration.log` records results.
- New Python code passed Ruff and syntax checks; both rank launchers passed dry-run verification.

No throughput/latency improvement is certified by these correctness checks. End-to-end inference smoke and live process verification follow startup.

## Rollback

Each node's prior config is in `deploy_config.before-r0.json` / `deploy_config.before-r1.json`; prior launchers are in `launch_rank.before.py` / `launch_rank.before-r1.py`. Restore the appropriate config and launcher on each node, then restart the head service. The old mode remains the default when the new option is absent. Historical experiment source files were not overwritten; active candidate copies are in this directory.

## Live acceptance, 2026-09-16

Both rank services are active with the new mode. Head process list contains API, resource tracker, EngineCore and GPU worker; worker-node list contains headless vLLM, resource tracker and GPU worker. There is no PLE subprocess. The effective custom seccomp policy and in-process environment flag were checked on both ranks.

Live smoke checks passed: health 200; unauthorized model access 401; both aliases; four concurrent arithmetic responses; tool call; streamed READY. Evidence: smoke.json. EngineCore affinity was restored to CPUs 15–19 after restart; GPU workers remain on 0–19.

A 20-second uprobe trace observed 139 successful native-reader calls on spark-1 and 150 on spark-2. Typical batch size was 1,024 rows (~16 active decode requests); median reader time was 3.116 ms and 3.782 ms, p95 3.983 ms and 5.092 ms respectively. This is not directly comparable to the earlier ~192-row/three-request trace and does not establish an end-to-end latency gain. Raw traces and live-reader-timing.json preserve the evidence. No runtime error/traceback was found on the head during acceptance.

The original 16-thread setting remains in the config for the legacy gather path; the new reader neither uses OpenMP nor calls that gather. Kernel io_uring workers still perform asynchronous I/O.
