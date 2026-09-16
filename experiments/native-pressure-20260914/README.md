# Native GPU cache with eviction-triggered disk backing

The previous completion policy allocated separate GPU snapshots out of the
native KV pool and staged resident restores through CPU memory. The sampled
snapshots occupied approximately 31 GiB per rank. This candidate removes that
policy: completed requests return their blocks to the normal native prefix
cache, without a completion snapshot or transfer.

## Serving policy

- MTP3; the deterministic kernels, batching, workload settings, 40 GiB GPU KV
  budget per rank, and 48 GiB logical disk budget per rank are unchanged.
- `disk_write_policy=pressure`, `memory_completion_cache=false`.
- Normal completion neither reserves a second GPU copy nor writes to disk.
- Normal native lookup/sharing and native partial-boundary replay apply.
- Only `BlockPool.get_new_blocks()` selecting a cached physical block for
  reuse creates a native backing-store job. Uncached/free blocks create none.
- The old bytes are saved before worker finish/free/add/update request hooks
  can zero, copy into, or otherwise overwrite the reused physical block.
- Every rank must acknowledge the save before the disk key becomes a hit.
- Existing pressure-driven active-request suspension/resume remains enabled.

The eviction hook uses the existing full native hash, translates the native
group ID into the connector group namespace, and stores the old physical page
under the same key the existing aligned external lookup computes. It does not
pin the page, allocate a replacement cache page, or change native LRU order.
Existing external hits avoid another store. Partial entries and disposable
compression rings are not misrepresented as full aligned external chunks.

`native_pressure.py` handles jobs separately from request-owned transfers so a
completed request need not remain in the scheduler. The existing transport
still performs disk content deduplication, checksums, and GPU verification.
The fence intentionally waits for the pressure-triggered save before reusing
its physical source; this does not serialize normal request generation.

## Validation and deployment artifacts

- `cpu-gates.log`: real block-pool and disk-slot ownership tests; no-pressure
  zero stores, no duplicate GPU allocations, two-rank acknowledgement,
  existing-backed-page reuse, same-step stale hash rejection, partial/ring
  exclusion, invalidation, and early worker ordering.
- `gpu-oracles.json`, `gpu-oracle-r0.log`, `gpu-oracle-r1.log`: actual isolated
  CUDA→disk→CUDA tests on both nodes. Source bytes were overwritten after the
  fence; restored bytes matched exactly and unrelated groups remained intact.
  These are transport/fence tests, not full-model eviction/resume tests.
- `before-r0.json`, `before-r1.json`: exact prior serving configs.
- `candidate-r0.json`, `candidate-r1.json`: deployed candidate configs.
- `runtime-verified.json`: all 20 overrides checked inside both containers.
- `start.json`, `readiness.log`: the single candidate restart and readiness.
- `probe_native.py`: serial–C10–serial token and logprob comparison with native
  prefix reuse, plus snapshot and global transfer counters. Ordinary workload
  traffic remains enabled, so global counters include it.

## Limits

Native caching retains its existing alignment/retention semantics: this change
does not promise an exact completion-boundary hit. A native unretained suffix
can be recomputed. Disk lookup likewise requires a complete aligned hybrid
prefix; retaining an isolated evicted page cannot guarantee a usable hit.
Disk backing is bounded: if all logical slots are pinned for suspended tasks,
ordinary native eviction may proceed without backing that page. No false hit
is published. There is no concurrency threshold; actual block reuse and active
request reservation pressure determine when offloading occurs.
