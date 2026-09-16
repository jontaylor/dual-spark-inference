# Batched page-prefetch PLE — rejected live candidate, 16 September 2026

**Final selection: mode0 (previous SQPOLL read path) on both ranks.**
The live comparison failed the predeclared GPU-readiness selection rule.
The candidate was faster in standalone warm/cold tests but delayed live graph
launch enough to regress GPU readiness. It is not promoted to canonical source.
The deployed experimental binary remains in fixed mode0; no alternation remains.
See `results/ple-resident-20260916/selection.json`.

Candidate reader preserves in-process ownership, GPU hashing, mapped output,
software row cache, exact deduplication and deferred GPU-graph overlap. For
software-cache misses it queues page-cache readahead with vectorized
`process_madvise(MADV_WILLNEED)` calls, then copies directly from its read-only
file mapping into output. Every vector is issued before the first mapped row
copy. It avoids per-row mincore calls and the read-to-scratch copy. Cold page
faults can still wait during completion, after the prefetch requests were issued.
This is advisory readahead, not a guarantee that the kernel submits every disk
operation simultaneously.

The original SQPOLL reader remains a fallback for unavailable/partial advice or
a detected file-size change. Advice failure can cause duplicate advisory/read
requests but never publication of an unchecked native read result. The packed
table must remain immutable while mapped, as with the original mmap gather;
concurrent file replacement/truncation is not supported. Disk errors during a
mapped fault can terminate the process, as in the original mmap path. The
existing truncated-file test falls back to io_uring and returns EIO without
publishing output. Pending io_uring buffers retain their existing ownership rules.

`GB10_PLE_READ_CONTROL` optionally names a shared 4096-byte control file. Its
aligned little-endian uint32 at offset0 selects0 = original SQPOLL issue,
1 = batched prefetch,2 = a64-step ABBA comparison (old/new/new/old). Default when
no file is configured is1. The launcher explicitly mounts the configured
`optimizations.ple_read_control` file read-only; the host can set the mode without
restarting or replacing frozen code. Production selection must end in a fixed
mode. The file lives outside the KV cache root, whose contents are validated.

Both nodes use `/home/jon/.cache/vllm-ple-control/ple-read-policy.bin`. The
container maps it at `/opt/gb10/ple-read-policy.bin`. The seccomp profile adds
only pidfd_open/process_madvise to the existing io_uring-enabled profile.

Validation evidence is under `results/ple-resident-20260916/`:

- Existing exact-byte, multi-ring, duplicate, collision, pending-output and
  truncated-file tests passed under normal and SQPOLL APIs.
- Real CUDA-graph byte checks, CPU/CUDA input staging,20 reuses, deliberately
  delayed publication, timeout telemetry and300 hash cases passed.
- `gpu-benchmark.json`: identical CUDA inputs, randomized method order,
 60 measured repetitions per method/shape after10 warmups; GPU events end after
 actual output-byte consumption. This synthetic warm test disables software
 cache. Original gather comparisons exclude IPC and use an already-resident
 CPU source, so they are optimistic native baselines, not end-to-end serving.
- `verified-cold.json`:24 repetitions per method/shape; every target page was
 verified nonresident before each measurement using mincore. All mappings of
 the separate test file were invalidated before fadvise; no live table was
 evicted. All outputs matched exact expected bytes.
- The earlier `native-benchmark.json` and `prefetch-benchmark.log` use advisory
 cold setup that does not invalidate every mapping. Do not call those verified
 cold measurements; use `verified-cold.json` instead.
- `live-ab/` and comparison JSONs: same-worker/cache ABBA, with actual prefetch
 success traced and both ranks aligned. Here comparator labels `async` mean
 batched prefetch and `inline` mean prior SQPOLL, not their usual meanings.

The first start attempt rejected the control file in the KV cache root before
model launch. The file was moved to the dedicated directory on both hosts and
both configs were installed before head start. No systemd daemon-reload was
performed. EngineCore and GPU workers remain unrestricted on CPUs0–19.

Rollback: write0 to both control files for the prior SQPOLL algorithm, without
restart. Full binary rollback uses the preserved `deploy.before-r{0,1}.json`
and `launch.before*.py`, installing both configs before head start. Never modify
a mounted shared library in place.

Mapped completion uses ordinary page faults; the io_uring completion timeout does not bound a stalled mapped fault. This is another limitation of the rejected candidate.
