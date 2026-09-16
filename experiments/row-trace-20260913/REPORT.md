# Server-side concurrency trace — 2026-09-13

The final model hidden states already differ before the output projection and
sampler. In this run, the first observed differences coincide with a mixed batch
containing one decode token and three 330-token prompt suffixes. The captured
outer request mappings, token positions, cache metadata and response attribution
are consistent. This localizes the next investigation to model computation or
cache-state handling within that forward pass; it does not identify a faulty
layer or kernel yet.

## Controls and reproduction

vLLM 0.29.0, image `vllm-gb10:v029-roce-qsa55122`, NVIDIA
Qwen3.8-Flash-Next-NVFP4 on two GB10 nodes with TP2+EP. Speculation remained off,
cache block size remained explicitly 1600, and the existing FLA override and
other inference settings remained in place.

A temporary, hash-verified model-runner override added opt-in snapshots before
model execution and around sampling. It copied tensors without modifying model
inputs or outputs. Capture was activated only for request IDs containing
`rowtrace-`, and limited to the first few generation steps. Both nodes were
restarted once to load the override. The capture helper passed a CPU self-test.

Replayed the archived 1930-token request with its original tools, thinking,
temperature-zero settings, neutral penalties, token IDs and top-5 logprobs.
Only the request ID was added; the final sequence also added seed 917352.
No tools were executed. Each sequence checked that the endpoint reported zero
running/waiting requests at phase boundaries.

| Sequence | Serial before | Concurrent | Serial after |
|---|---|---|---|
| Capture inactive | 3 identical | 4 distinct | 3 identical |
| Capture active | 3 identical | Same set of 4 outputs | 3 identical |
| Capture active, fixed seed | 3 identical | Same set of 4 outputs | 3 identical |

All 30 requests succeeded. All 18 serial responses matched. The client ordering
of concurrent outputs varied, but the output set was identical across all three
sequences. The first uncaptured request reported zero cached prompt tokens; the
other 29 reported 1600. Capture therefore preserved the observed divergence in
this control, despite its synchronization/timing overhead.

The serial output differs from the earlier remote report, including changed
score margins. This experiment uses its own post-restart serial control; it does
not claim to reproduce the earlier 6.125-unit reversal exactly or explain the
change across restarts. Client transport was Python urllib on the head node,
instead of httpx on the orchestration host.

## Earliest observed difference

The first concurrent request ran its 330-token suffix alone. Its hidden state
and complete raw logit vector for generated token 1 were bit-identical to the
serial control. The next scheduled batch contained:

```text
batch rows:              0      1      2      3
computed tokens:      1930   1600   1600   1600
scheduled tokens:        1    330    330    330
generation prediction:  2      1      1      1
total forward tokens: 991
```

The model's final hidden states and raw logits now differed despite identical
per-request input prefixes. Differences relative to the corresponding serial
generation step were:

| Batch row | Predicted token number | Hidden relative L2 difference | Maximum absolute raw-logit difference |
|---|---:|---:|---:|
| 0 | 2 | 0.307731 | 2.945313 |
| 1 | 1 | 0.755735 | 10.125000 |
| 2 | 1 | 0.789694 | 10.625000 |
| 3 | 1 | 0.739949 | 10.640625 |

The selected first two tokens still agree (`Let`, ` me`). At prediction 3,
the serial raw-logit gap `look - investigate` is +2.25. Concurrent rows have
gaps -1.25, -5.375, +3.5, and +0.125. Rows 0 and 1 choose ` investigate`;
the others choose ` look`. All comparisons at predictions 1–3 have identical
preceding token IDs.

These are measured batch rows, not inferred client submission positions.
The next batch is four single-token decodes, but it already inherits the
different states produced by the mixed pass. Its divergence alone cannot
implicate the decode-only graph.

## Attribution and cache checks

Captured 72 forward/sampling steps per node, comprising 102 request-generation
rows per node. The trace contains padded input arrays as well as actual row
boundaries; padded stale token values were not treated as real request inputs.

- CPU/GPU request indices, forward/reverse request maps and sampling request
  IDs agree in every captured row.
- Input token IDs match the request-state token buffer at the recorded positions.
  Positions are contiguous and agree with computed-token counters and sequence
  lengths. Effective CPU and GPU temperatures are zero.
- Every sampled token is a maximum of its captured pre-sampling raw logits.
  The API token IDs agree with the captured selection in all 102 head-node rows.
  Recomputed selected-token logprobs agree with the API within 4.24e-6.
- Batch-sharded sampling is disabled in the captured runner.
- All 612 generic slot-mapping checks pass. The circular-buffer group correctly
  uses the disabled-mapping sentinel -1; its backend handles its own ring.
- No physical writable-block overlap was found between concurrent requests in
  the captured block tables. Shared cached prefix blocks are distinct from the
  current writable blocks. Mamba metadata state indices are consistent with the captured Mamba
  cache-group tables at each request's logical state index.
- Both TP nodes have bit-identical captured input tokens, positions, final
  hidden states and raw logits across all 72 corresponding steps.
- The fixed-seed repeat has bit-identical hidden states and logits for all 51
  corresponding captured rows, indexed by phase, batch row and generation step.

These checks support correct attribution at the runner/sampler/API boundaries.
They do not prove correct internal layer indexing, actual kernel memory accesses,
or correct KV/recurrent-state contents. Layer-to-cache-group assignment was
not independently verified. Cache values and layer-by-layer activations were
not captured. Cross-node equality does not rule out a
deterministic tensor-parallel computation error shared by both nodes.

## Next diagnostic

Capture layer boundaries and relevant initial recurrent/cache state for the
991-token mixed forward, comparing its decode row and each prefill suffix with
the corresponding isolated run. Locate the first differing layer before
changing kernels or disabling optimizations. A separate control that prevents
prefill/decode mixing would help test whether mixing itself is necessary.

## Service state and evidence

Capture controls were removed from both live containers. Both saved deployment
configurations exactly match their pre-capture versions. The running diagnostic
modules remain loaded but inactive until the next restart, which will load the
original runner. No second restart was needed. Health returned HTTP 200 after
cleanup. Speculation remains disabled.

Files in this directory:

- `request.json`, `reference-results.json`: archived request and earlier remote results.
- `uncaptured/`, `captured/`, `captured-seed/`: requests, raw SSE, assembled
  responses and boundary metrics; `api-summary.json` summarizes the controls.
- `traces-r0/`, `traces-r1/`: raw tensor snapshots and `analysis.json`.
- `traces-r0/checks.json`: cache, TP, API attribution and seed checks.
- `row_trace.py`, `model_runner.py`: exact diagnostic helper and runner override.
- `config-before.json`: head-node deployment configuration before instrumentation.
  The corresponding worker backup remains at the same path on the worker.
- `probe.py`, `analyze_trace.py`, `check_trace.py`: reproduction and analysis code.
  Tensor analysis requires PyTorch; it was executed in the running vLLM container.
- `evidence-sha256.json`: artifact hashes.

