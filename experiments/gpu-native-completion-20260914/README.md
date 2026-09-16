# GPU completion reuse with native allocator lifetime

Candidate v2 restores completion endpoint reuse without periodic decode copies,
CPU state staging, duplicate attention history, or permanent GPU reservations.
MTP3, concurrent inference kernels, physical 1920-token pages, 40 GiB KV/rank and
48 GiB disk/rank remain configured. Recurrent retention is semantic (`0`).

At completion, attention and ring pages are shared from the finishing request;
one normalized recurrent page per group is allocated from the ordinary pool.
One batched GPU kernel selects accepted speculative state and shifts convolution
history. Only a small validity vector returns to CPU. Both workers acknowledge
the event before its endpoint is published. The pages then enter the ordinary
free/LRU pool; they consume no permanent reservation credits.

The allocator recognizes indexed completion pages as cached even when they lack
a coarse native hash. Thus unneeded blocks are reused before recent completion
state. A selected completion page is saved to disk before worker request updates
can overwrite it. Disk checksums, deduplication and restore readback remain on.
Save acknowledgement removes old metadata without freeing the block from its
new owner. Intermediate native hashes do not initiate disk writes in this mode.

A lookup pins the nonlocal source pages across destination allocation. A load
then owns those pins until it completes. An unscheduled lookup releases its pins
at the end of the scheduling pass. This closes the lookup/allocation race found
in final review of v1; that startup was stopped before live validation.

GPU completion recalls copy only missing tail/state pages directly on GPU.
Already adopted native prefix pages are shared. A pressure suspension creates
a disk checkpoint; it does not inherit GPU-only backing and pin away the memory
that suspension must reclaim.

## Verification before live validation

- `ownership-v2.log`: actual allocator/slot manager, no permanent reservations,
  source-load pins, backing only on physical reuse, both-worker ACK gate,
  no freeing of the new block owner, failed-store and retirement cleanup.
- `finish-plan-v2.log`: actual completion planner, two turns share three attention
  pages and allocate one state page per turn; suspension retains no GPU backing.
- `gpu-roundtrip-r0.log`, `gpu-roundtrip-r1.log`: real GPU event capture and recall
  without disk traffic, forced source overwrite after the early eviction fence,
  exact disk restore with GPU readback on each node. Uses isolated small pages.
- `lookup-lifetime.log`: source survival across destination allocation, handoff
  to the actual load, and release for unscheduled waiters.
- Prior full model-size copy oracle: `../checkpoint-sweetspot-20260914/`.

These gates validate components and ownership, not full-model determinism or
aggregate throughput. `probe_live.py` performs the subsequent C1/C4/C10
token/logprob and endpoint-reuse checks and a synthetic concurrent multi-turn
throughput test. Live results must be recorded before claiming a deployed fix.

## Configuration and recovery

`before-r0.json` and `before-r1.json` preserve the prior serving configurations.
`candidate-v2-r0.json` and `candidate-v2-r1.json` contain all final override paths
and checksums, including immutable `completion.gpu.v2.py`.
`start.json` records v1; `start-v2.json` records the corrected candidate.
The launcher passes `native_completion_cache` alongside the existing completion
event flag. No baseline-only reload was performed.

The connector prefix-hit counter includes GPU-resident completion hits: it is
not inherently a disk-hit counter. Actual disk load/store byte counters exclude
GPU-only copies. Interpret those counters together when assessing the cache
path; do not describe a GPU completion recall as a disk transfer.
