# Completion reuse regression and checkpoint work

## Finding

Removing completion-state reuse exposed a large native replay penalty. The
"earlier warm" campaign spans that deployment; comparing whole campaign averages
conceals the change within the campaign itself.

The frozen trace contains 502 matched warm follow-ups and 371 overlapping
follow-ups (the overlapping campaign continued after the user's 226-row sample).
Splitting the warm campaign around the recorded deployment gives:

| Warm campaign follow-ups | Before restart | Both requests after verified native start |
|---|---:|---:|
| Requests | 315 | 162 |
| Mean uncached input | 1,239.5 | 3,838.2 |
| Estimated newly added input | 1,011.8 | 570.6 |
| Estimated existing content replayed | 227.7 | 3,267.6 |
| Hits at estimated previous computed endpoint | 295 | 0 |
| Hits aligned to 1,920 tokens | 1 | 162 |

Twenty-five transition rows are excluded from this phase comparison. The
restart timestamp is 1789387596.7642553; the conservative post-deployment cutoff
is the verified native audit timestamp, 1789388359.5779467. Both the predecessor
and follow-up must pass the latter cutoff. This is observational evidence, not
a paired throughput experiment. It identifies a concrete cache regression;
it does not attribute every throughput change to that regression.

Source inspection corroborates the mechanism:

- `memory_completion_cache=false` disables completion endpoint reuse.
- Native physical and match blocks are both 1,920 tokens.
- The MTP attention lookup drops a further match unit.
- Existing fine-grained Mamba partial-tail publication targets the original
  prompt, not the generated response endpoint.
- Merely enabling fine hashing therefore does not restore generated-state
  reuse, and the Mamba state must exist at the boundary allowed by MTP lookup.

The deployment fixed the duplicate-memory/CPU-copy behavior but failed to
replace completion reuse with an efficient native mechanism. That tradeoff
introduced the observed replay regression.

## Measurement definitions and audit

`extract_trace.py` matches the longest earlier exact wire-message prefix within
each campaign arm. It does not assume the preceding request number is the
parent: worker branches interleave. All 873 matched rows also contain the prior
assistant output with the same content, reasoning, function names and parsed
JSON arguments. That check ignores tool-call IDs, which the harness sometimes
renames. The harness also reserializes JSON arguments. This is a semantic output
check, not proof of exact generated-token identity or unchanged template/tool
rendering.

The replay estimate uses previous prompt + generated tokens - 1 as the previous
computed endpoint. The one-token allowance is why estimated new input differs
by one token from the workload task's `growth - previous_completion` calculation.

The table's older-prompt floor is `max(previous_prompt - cached, 0)`, calculated
per request and then averaged. It is not `mean(uncached) - mean(prompt_growth)`:
the latter permits requests with good output reuse to cancel older-prefix losses
elsewhere. The warm 1,877.6-token floor is reproduced exactly.

The later 371-row overlap sample averages 4,330.6 uncached tokens, 900.7 estimated
new tokens, and 3,430.0 estimated replayed tokens. All hits are 1,920-aligned.

## GPU copy implementation and actual tests

`gpu_checkpoint.py` implements an experimental single-launch batched copy of
accepted recurrent state. It selects the temporal speculative state and shifts
the convolution time axis using GPU computed/accepted/column metadata. It copies
bytes without floating-point arithmetic. It neither copies attention history
nor stages state through CPU memory. It is **not installed in the server**.

`check_gpu_checkpoint.py` uses the model's actual MTP3/TP2 shape calculators:
36 GDN layers plus the PLE convolution, totaling 59,080,704 bytes (56.34 MiB) per
request per rank. It checks both convolution axis layouts, accepted counts 1–4,
nonzero source columns, valid rollback and unavailable state, missing blocks,
out-of-range columns, and CUDA graph replay. Invalid state must not modify the
destination. GPU allocations are isolated from live KV memory; no model weights
are loaded.

The first oracle passed on both ranks. Under background inference, initial GPU
event measurements were approximately 5.4 ms for one request on each rank and
24.0/12.5 ms for one four-request launch on ranks 0/1 respectively. The samples
were not simultaneous or controlled for background contention; they are not a
two-rank critical-path measurement or a model throughput result. Full timing
samples are retained in `gpu-copy-r0.log` and `gpu-copy-r1.log`. Expanded oracle
results are retained separately in `gpu-copy-final-r*.log`.

The runtime must still provide safe destination allocation, source lifetime,
both-rank completion acknowledgement, native hash publication, rolling-state
replacement, and reuse/eviction integration. A correct copy primitive alone
does not establish correct model continuation.

## What interval estimates do and do not establish

The frozen overlap trace projects the following if an appropriate generated-state
checkpoint always survives and is usable:

| Checkpoint interval | Mean arithmetic remainder | Decode checkpoint crossings per previous response |
|---|---:|---:|
| 64 | 31.2 | 15.62 |
| 128 | 63.0 | 7.82 |
| 256 | 135.8 | 3.87 |
| 512 | 269.6 | 1.91 |

These are projections, **not measured candidate replay or throughput**. They
omit MTP's extra matching backoff, state availability, eviction and template
token changes. With a 64-token matching unit, the checkpoint chosen for MTP may
need to precede the final interval boundary; a predecessor state must actually
be retained. Adding 64 to a mean remainder is not a substitute for simulating
that selection.

Physical block size, hash granularity and checkpoint cadence are separate.
The installed hash resolver requires the match unit to divide 1,920: 64 and
128 qualify, while 256 and 512 do not. A 256/512 checkpoint cadence can instead
use 64-token hashes and remain on the deterministic 64-token arithmetic grid.
Changing physical blocks just to accommodate a cadence would change memory
geometry and confound the comparison.

No empirical interval winner has been established. Completion-triggered reuse
also remains a candidate: one event copy can be cheaper than periodic copies,
but exact-boundary lookup, speculative normalization and QSA/GDN continuation
must be validated together. Do not infer that periodic 64-token copying is the
required design from the interval projection.

## Serving status

No serving configuration or runtime override was changed, and no model restart
was performed during this work. MTP3 and concurrent batching remain configured.
After the isolated copy tests the service was active and `/health` returned 200.
The native-cache integration, deterministic continuation tests and representative
aggregate-throughput sweep remain unfinished. This report does not claim a
deployed performance fix.

Reproduce the phase split with `python3 analyse_phases.py`. Source inputs and
intermediate audits are retained; `trace-final.json` is the frozen final trace
for this report and `phase-analysis.json` contains the numerical results.
