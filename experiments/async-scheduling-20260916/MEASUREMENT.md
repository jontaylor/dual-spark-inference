# Synchronous scheduling: live exposure measurement

16 September 2026. No serving configuration, affinity, model, workload or
runtime source changes. No restart. Both bounded captures have completed.

The statement above describes the measurement windows. Subsequently the
synchronous service crashed independently during same-pass park/resume at
17:35:20 UTC. A narrow fix was applied and recovery started; see
`../same-step-parking-resume-20260916/REPORT.md`. Async remains uninstalled.

## Result

### Admission audit (18:52 UTC update)

The 9 ms lower-concurrency result below covers selected host handoffs only.
It is not a bound on total synchronous scheduling cost or possible throughput
benefit. The deployed `step_with_batch_queue` prioritises submitting another
step before receiving the oldest result; `AsyncScheduler` uses outstanding
output placeholders to permit this. Neither mode inserts requests into an
already executing forward pass. Continuous batching operates at step boundaries
in both modes.

Existing campaign telemetry also exposes admission pressure. After recovery,
at 9–16 running requests, 48/152 approximately five-second samples had waiters;
47 had deferred waiters and 6 had capacity-labelled waiters (categories overlap).
At 1–8 running requests, 11/78 samples had waiters. These are sample counts,
not per-step durations or ready-work counts. Reproduce with `analyze_admission.py`;
source and output are `campaign-backend-metrics.jsonl` and
`admission-summary.json` under the results directory.

Source verification matters: `capacity` is simply `len(self.waiting)` and
`deferred` is `len(self.skipped_waiting)`. They are queue membership labels,
not complete causal diagnoses. Our deployed parking scheduler enforces strict
waiting order. A blocked head request can stop admission of subsequent requests;
quiescing also blocks new admission. Async scheduling alone preserves these
rules, so it does not automatically remove their under-utilisation.

The paired low-concurrency capture included completion restores, not just steady
decode. At 18:34:14 the server log reports 13 running and two deferred requests;
restore jobs 6193/6194/6195 were verified at 18:34:11/13/14. Transfer metrics in
that interval include substantial restore work. These observations establish
additional work/dependencies during the capture; summed transfer durations are
not GPU idle durations. Source: `low-capture-server.log`. The six Prometheus
samples during the capture missed these deferred requests, demonstrating the
sampling limitation directly.

The campaign owner confirms that this particular client did not record per-request
first-token receive times. Its input-file timestamps approximate client dispatch,
not server arrival. They cannot resolve millisecond admission delays. Actual
per-step readiness/dispatch and GPU kernel attribution remain outstanding.

The visible synchronous host handoff is largely overlapped by GPU activity in
these captures. It must not be counted wholesale as GPU idle time. There is
not yet evidence supporting an immediate async deployment for a large speedup.
This does not establish that async scheduling has no benefit.

The live engine uses `EngineCore.step`, which schedules, dispatches, waits for
output, updates the scheduler and repeats. However, the worker's `AsyncOutput`
records its output-copy event **before** `postprocess_sampled` and the MTP
speculator. The worker enqueues the drafter work before returning the output;
waiting for that output event does not wait for all drafter GPU work to finish.
That work can overlap engine output processing and the next scheduling call
even with `--no-async-scheduling`.

## Measurements

### Natural drain: 1–2 requests (19:02 UTC update)

Two further paired captures completed without serving changes. In the first,
starting 18:54:44.944 / 18:54:45.915 UTC, selected host handoffs averaged
2.694 / 3.181 ms (339/340 intervals), with no zero-SM samples within them.
Whole-capture zero-SM fractions were 1.555% / 1.774%. The longest spans were
155.8 / 173.6 ms, around 18:55:07, during a one-to-two-request transition
with checkpoint restore. These spans were outside the selected handoffs;
the capture does not identify their exact blocking operation.

The follow-up, starting 19:00:14.721 / 19:00:15.773 UTC, added CUDA API
entry/return probes for waits and copies lasting at least 1 ms. Main-thread
`cudaStreamSynchronize` occupied 2.859 / 2.327 seconds of each 30-second
capture, but contained only 13 / 23 zero-SM samples (about 1.3 / 2.3 ms).
`cudaEventSynchronize` occupied 19.116 / 18.456 seconds, with only 5 / 0
zero-SM samples. Thus these measured host wait durations predominantly
overlap GPU activity, rather than representing exposed idle time. This does
not distinguish useful compute from device-side communication waiting.

The follow-up's largest zero-SM span was 3.608 seconds on both ranks.
Timestamped server logs show the last request completing at 19:00:33.441,
zero running and waiting requests at 19:00:34.221, and the next admission at
19:00:37.059. The span was 19:00:33.454–37.062. This is strong evidence of
workload starvation during most of that interval, not a scheduler stall.
The remaining zero-SM samples total approximately 119 / 241 ms across the
30-second captures, including transitions and copies; they are not established
as recoverable scheduling cost.

An intervening 45-second worker stack sample again found the blocking metadata
copy at `short_conv_attn.py:333` prominent (4.885 of 43.09 sampled seconds), but
this was not simultaneous with the successful wait capture and cannot be used
to assign every stream wait to that call site.

Evidence: `tail-concurrency/`, `tail-waits/`, `tail-waits-fixed/` under results.
`tail-waits` is a failed combined-probe attempt: duplicate END probes prevented
attachment. Its separately attempted wait probe produced no records because
function addresses were unsuitable entry/return keys. Do not use it for a
negative finding. Both issues were corrected before `tail-waits-fixed`, which
completed on both ranks with valid wait records. `analyze_waits.py` reproduces
the latter's API-wait overlap results. All observers and profilers have exited.

CUDA runtime uprobes recorded main-thread event synchronization completion;
driver uprobes recorded its next kernel/graph launch. GPU hardware metrics
were captured independently with Nsight Systems at 10 kHz, without attaching
CUDA tracing or restarting the process. Each host's wall/monotonic clock offset
maps the probe timestamps to its own Nsight capture. Captures were sequential,
not simultaneous, and their steps must not be treated as matched pairs.

| Measurement | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| Capture start UTC | 17:11:01 | 17:13:36 |
| Duration | 30 s | 30 s |
| Long-event-return → next-launch intervals | 109 | 100 |
| Mean host interval | 5.78 ms | 6.07 ms |
| Median | 5.24 ms | 5.18 ms |
| p95 | 10.18 ms | 11.34 ms |
| GPU GR Active during host intervals | 99.87% | 99.83% |
| GPU SMs Active during host intervals | 88.31% | 85.23% |
| Zero-SM samples during those intervals | 0 | 0 |
| Whole-capture GR Active | 99.28% | 99.11% |
| Whole-capture SMs Active | 84.46% | 82.16% |
| Whole-capture zero-SM sample fraction | 0.139% | 0.184% |
| Longest consecutive zero-SM sample span | 2.30 ms | 1.70 ms |

The main table selects event waits of at least 50 ms to separate long output
waits from persistent PLE input-copy/consumption events. It is a heuristic,
not Python-call-site identification. Thresholds of 10 and 100 ms are also
reported in the machine-readable summary. Kernel launch probes cover
`cuLaunchKernel`, `cuLaunchKernelEx`, and `cuGraphLaunch`. They do not capture
every CUDA API. Copies and other work may occur before the next launch.

The observed host intervals total 630 and 607 ms respectively. Their hardware
activity does not show an idle bubble of that size. GPU activity also does not
prove that every operation was useful: collectives and device-side polling can
count as active. Average SM issue was only 5.60% / 5.27% across the captures;
that metric is not a direct compute-efficiency or async-speedup estimate for a
memory-intensive model. Tensor-core inactivity likewise includes legitimate
non-tensor work. A causal throughput improvement requires a comparable async
implementation and comparison, or a complete kernel/dependency timeline.

The workload is now a different, finite 24-condition campaign,
`20260916T165454Z-baseline1609-factorial-24`, with around 29–30 server/client
requests near the measurements. These findings do **not** establish exposure
at the earlier 11–12-request workload. The workload owner confirmed retry
support for a bounded restart; no restart has been requested.

## Reproduction and limits

`host-gap.bt` is the bounded rank-0 probe; the raw rank-1 variant uses its
worker PID and namespace library path. PIDs must be revalidated before reuse.

Hardware collection:

```bash
sudo -n nsys profile --trace=none --sample=none --cpuctxsw=none \
  --gpu-metrics-devices=0 --gpu-metrics-frequency=10000 --duration=30 \
  --output=CAPTURE
nsys export --type=sqlite --output=CAPTURE.sqlite CAPTURE.nsys-rep
.venv/bin/python experiments/async-scheduling-20260916/analyze.py
```

Raw evidence is in `results/async-scheduling-20260916/`, including Nsight
reports, SQLite exports, probe rows, clocks and `summary.json`. Probe and metric
overhead was not independently A/B quantified. Metrics are quantized and
sampled, not exact kernel start/end traces. The reported longest zero spans
are spans between sample timestamps, not exact idle durations. GPU occupancy
and useful inference throughput must not be conflated.

## Compatibility finding

The exact deployed completion method skips snapshots while
`request.num_in_flight_tokens != 0`. With async lookahead, ordinary termination
can occur in that state. Merely draining the next batch does not solve it:
the GPU recurrent state may have advanced beyond the accepted terminal
boundary. The native snapshot kernel rejects a boundary outside its current
speculative window. Accepted-token-only parking reservations also omit
outstanding output placeholders.

`audit_async_boundaries.py` exercises the actual completion guard without
loading model weights and checks concrete boundary/reservation counterexamples.
It passes as an incompatibility audit, **not** an async-readiness test.

Next work: finish the source-level async patch and its state-ownership tests;
obtain comparable lower-concurrency evidence before inferring a general
performance benefit. The current evidence does not justify removing the
synchronous guards and deploying immediately.

## Subsequent candidate-only measurements

Isolated device-state checks passed on rank 0 at 18:13:30–18:13:38 and
18:16:26–18:16:33 UTC, with tiny independent buffers and no serving code changes.
The candidate marker plus nonterminal conditional-copy path measured median
24.08 us at 12 requests and 51.78 us at 32 requests (p95 40.93 / 87.58 us),
30 samples each, 86 descriptors/request. `candidate-overhead.json` records the
exact interval and samples. These CUDA-event spans include contention and
host launch spacing; they measure added candidate cost, not the throughput
benefit of async scheduling. The workload owner was informed of interference
intervals for exclusion from strict comparisons.

The first lower-concurrency observer expired at 18:11 UTC without triggering.
The owner reports recent concurrency 20–29 and no reliable natural-drain ETA.
A single new observer waits up to an hour from 18:20:43 UTC; no load was changed.

## Completed lower-concurrency capture

The second observer triggered after two 14-request observations. Both ranks
were captured concurrently for about 30 seconds: rank 0 started 18:33:59.953
UTC and rank 1 started 18:34:01.016 UTC. Running-request samples during capture
ranged 12–16; this is a lower-concurrency window, not a fixed 12-request test.
No isolated candidate GPU test overlapped this capture (the preceding one
finished at 18:27:05.712; the following prototype started at 18:36:30.586 UTC).

| Metric (50 ms output-wait heuristic) | Rank 0 | Rank 1 |
| --- | ---: | ---: |
| Host handoffs | 61 | 63 |
| Mean host handoff | 6.004 ms | 6.708 ms |
| Median | 5.279 ms | 5.937 ms |
| p95 | 9.777 ms | 11.860 ms |
| Total host handoff time | 366.216 ms | 422.580 ms |
| GR Active during handoffs | 96.894% | 97.381% |
| SMs Active during handoffs | 87.037% | 86.492% |
| Zero-SM samples during handoffs | 94 | 91 |
| Approximate zero-SM time during handoffs (10 kHz) | 9.4 ms | 9.1 ms |
| Whole-capture zero-SM fraction | 0.171% | 0.203% |
| Longest zero-SM sample span anywhere in capture | 9.100 ms | 10.401 ms |

Most host handoff time remains overlapped with GPU activity. The explicitly
observed zero-SM portions of these handoffs occupy approximately 0.03% of the
capture, not the 1.2–1.4% occupied by the complete host intervals. This is **not**
an upper bound on possible async benefit: GPU activity can include communication
polling/waiting, and the output-event identification is heuristic. Kernel-level
attribution and a causal throughput comparison remain separate requirements.
Raw data and recomputation: `results/async-scheduling-20260916/lower-concurrency-second/`.
The observer completed and is no longer running; do not start a duplicate.
