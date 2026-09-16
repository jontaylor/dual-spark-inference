# Asynchronous persistence after native eviction source preservation

Candidate changes exactly two existing overrides and adds one helper (23 total):
rank_local_disk.async.py, connector.async.py, and pipeline.py mapped to
v1/kv_offload/gb10_staged_evictions.py. Before/candidate rank-specific JSONs and
runtime-verified.json preserve identity. MTP3 and numerical sources unchanged.

For native one-page DiskSlots eviction jobs only, submit a stage task to a
separate copy worker and bounded eight-page CPU region (217,251,840 bytes/rank
at current page size). Disk writes and hashes use the existing serialized disk
executor. Source-fence wait ends after DMA completion and enqueueing persistence.
The normal get_finished ACK waits for persistence, preserving existing scheduler
HIT_PENDING state and both-rank completion. Other transfer types and ordinary
wait calls retain full completion semantics. More than eight outstanding pages
causes backpressure until a buffer is released. No unbounded payload allocations.

Staging does not itself publish readable disk content, dedup hashes, or store
success. Copy errors fail both futures. Persistence errors fail the public job
and propagate through get_finished; no successful ACK is invented. Shutdown
first drains staging while persistence is still running, then drains persistence
and closes both copy regions. Existing checksum/load/GPU readback is retained.

The source-fence context is scoped around the existing handle_preemptions call
and removed in finally. Once source bytes are preserved, callers may overwrite
GPU blocks while old backing remains pending. This relies on the existing
native-pressure manager pinning storage keys (not reused GPU pages), rejecting
spilling keys with HIT_PENDING, and freeing only old metadata after all ACKs.

Checks:
- check_pipeline.py: held disk write, source completion before persistence,
  one-buffer backpressure, copy/write failures, buffer recovery.
- check_gpu.py: real 27,156,480-byte pages on each GPU; hold filesystem store,
  release source fence, overwrite GPU source, persist then restore byte-exact
  with checksums/readback; 20 source generations across eight reusable buffers.
- Live C1/C4/C10 and post-pressure stall capture: see live result artifacts.

Not claimed: cold/cache numerical parity, crash durability/restart recovery,
all pressure interleavings, or a throughput improvement before live measurement.
The code does not batch many eviction jobs into one DMA call; it pipelines
individual canonical page copies against hashing/persistence. Further DMA
batching is a separate optimization.
