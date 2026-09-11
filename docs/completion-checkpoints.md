# Completion checkpoints

The `kv_paging.completion_checkpoints` option adds an exact-position checkpoint
when a text request finishes between the existing 1,600-token boundaries. Active
checkpoint cadence, the aligned prefix, C32 admission, and the 262,144-token
context limit are retained.

A later request must have the same token prefix and cache salt. The record also
includes the following-token witness needed by MTP. The last emitted token is
normally unprocessed, so a matching follow-up replays that one token plus its new
input. The longest matching available checkpoint wins.

## Implementation

The scheduler's existing completion hook runs before GPU block release. The GB10
connector keeps those blocks and their reservation until all outstanding stores
have completed on both TP ranks. Existing aligned disk pages are reused where
available; private completion keys store the remaining pages.

The worker captures state before removing the request slot, after the previous
GPU work has settled. The snapshot covers attention KV, MTP attention, QSA raw
compression rings including position metadata, GDN temporal and convolution
state, and PLE convolution state. The existing convolution/temporal copy semantics
select the accepted speculative state. Normalization writes the CPU disk staging
page, without modifying live GPU state. PLE n-gram context is regenerated from
the restored token history rather than saved as an independent mutable cache.

This uses the existing disk slot allocator, 48 GiB per-rank budget, four staging
slots, checksums, readback verification, and all-rank acknowledgement protocol.
There is no separate unbounded completion directory or new inference kernel.

The index is capped at 256 records. Published pages remain evictable under the
ordinary disk LRU; lookup checks every dependency and falls back to aligned reuse
if any are missing. Candidate token prefixes are hashed in a single pass. An
engine restart expires these records along with the existing process-local cache.

Aborts and in-flight requests are not saved. Embedding-input, multimodal, LoRA,
and cache-bypass requests are excluded. Already-aligned completions use the
existing path. If output truncation selects a recurrent position already discarded
by speculative normalization, the exact snapshot is not published. Disk corruption
continues to fail closed through the existing transport.

## Validation

- 26 focused tests cover state selection in SD/DS convolution layouts, speculative
  truncation, delayed publication, salt/witness mismatch, longest-prefix selection,
  capacity failure, eviction, and real CUDA/disk round trips.
- Repository pre-commit hooks and four deployment lifecycle tests pass.
- Both ranks verify restored GPU bytes against stored checksums.
- Live follow-ups restore 2,916/2,917, 6,018/6,019, and 11,034/11,035 prompt tokens.
  Each replays one token. Their corresponding nearest aligned boundaries would
  leave 1,317, 1,219, and 1,435 tokens to replay; the ordinary MTP lookup can be
  more conservative still.
- Four concurrent follow-ups each reused 11,034 tokens and processed 901 tokens
  (900 new tokens plus the final old token), crossing the next aligned boundary.
  A changed final witness token rejected the exact snapshot and reused the
  existing aligned cache at 8,000 tokens.
- All three longer-tail tests matched 32 greedy cold-control tokens. One earlier
  test differed after 25 tokens; repeated cold and uninterrupted controls also
  varied, including differences before the checkpoint position. These checks
  validate reuse and state transport, not general bit-exact model determinism.
- The final rebuild also restored positions 1,599 and 1,602 with one-token replay
  and matched 32 cold-control tokens in both cases. Its 11K case again restored
  11,034/11,035 tokens; greedy output diverged after 10 tokens. Numerical
  equivalence across different execution shapes is not certified by these tests.

Aggregated evidence is in `completion-checkpoint-validation.json`; private raw
responses and scripts are kept outside this repository in the local experiment
folder. Logs include `Completion checkpoint saved`, `Completion checkpoint
restore`, and `Completion checkpoint skipped`, without token content.

## Operations

`prepare_paging.py` freezes the committed connector, completion helper, disk
transport, and model-runner hook into the existing hash-verified overlay bundle.
Both nodes use identical source bytes. The feature is enabled in the deployment
example and local configuration. To disable it, set
`kv_paging.completion_checkpoints` to `false` on both nodes and restart the head
service; active-request paging continues using aligned checkpoints.
