# Native pressure cache — deployed result

Implemented and left serving on both nodes. Health 200, service active, MTP3.

## Behavior

Completed requests now return their GPU blocks to normal native prefix caching.
There is no proactive completion snapshot, second GPU allocation, or resident
GPU→CPU→GPU connector restore. Native cached blocks are written to the backing
store only when the allocator actually selects their physical block for reuse.
The worker fences that save before any request-update/zeroing/COW operation can
overwrite the old bytes. Existing pressure-driven active suspension and resume
remain enabled. Full CPU checksums, content deduplication and GPU readback
verification remain enabled on disk transfers.

## Evidence

- One candidate restart; no baseline-only reload. Both configs and all 20 runtime
  overrides verified in the running containers.
- Ten concurrent follow-ups and one subsequent serial response matched the
  serial reference exactly in generated token IDs and returned logprobs.
- All follow-ups reused the same 3,840-token native boundary. The source prompt
  had 7,264 tokens; the follow-up had 7,396 tokens. Native MTP lookup rounds down
  and drops one further 1,920-token aligned block. The first probe assertion
  incorrectly assumed a one-block suffix; source inspection corrected the test,
  and its original failure is retained. Serving code did not change afterward.
- Server disk-store bytes: **0**; disk-load bytes: **0** throughout that live
  probe, including ordinary background traffic. Snapshot blocks: **0**.
- CPU allocator/ownership/order gates passed.
- Real isolated CUDA→disk→CUDA eviction tests passed independently on both GPUs:
  save via the new early fence, overwrite the source, restore exact bytes, and
  verify that unrelated groups were unchanged. Checksums and readback enabled.
- Final logs contained zero completion-snapshot plans, zero native eviction saves
  in this below-pressure interval, and no error/traceback lines on either node.
- A following 30-second ordinary workload sample averaged
  **136.9 aggregate output tokens/s**, with mean
  **8.0 running requests**. This is not a matched historical
  throughput comparison; earlier probe timing included large cold prefills.
- Final sampled active/pinned KV usage: **23.9%**.
  Zero-reference native cached prefixes are reusable and are excluded from that
  occupancy gauge; they remain physically on the GPU until normal reuse.

## Scope and limits

No new full-context C20 suspension stress run was performed. Active parking
code and its transport are unchanged. The new eviction fence was validated with
actual disk transfers on both nodes, but this is not a full-model disk-restore
oracle. Native alignment/replay semantics remain: an unretained tail can be
recomputed. A disk hit requires all needed hybrid groups, not an isolated page.
Disk backing is bounded; pages may be evicted without backing if all logical
slots are pinned. The trigger is physical cached-block reuse, not an active-only
occupancy percentage or a fixed concurrency threshold. Thus cached history can
require eviction even when active requests alone would fit. Cache counters
include the cold restart and synthetic correctness traffic.

The existing representative campaign was observed still running at
`20260914T113242Z-temperature-warm-repeat`; its source, sampling, concurrency,
cadence, and retries were not changed. Operator inference probes are finished.

See `README.md`, `FINAL-AUDIT.json`, `gpu-oracles.json`, and
`probe-1789388074910388442/summary.json` for implementation and raw evidence.
