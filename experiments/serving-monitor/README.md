# Passive serving monitor

Running as the transient user unit `ple-serving-monitor.service`. It does not
restart or reconfigure inference and does not generate benchmark requests.
The load-pause request was cancelled; the representative workload is left running.

Live output: `results/serving-monitor/latest.md` and `latest.json`, refreshed
approximately every five seconds. Timestamped JSONL observations are retained
for 24 hours. The rolling view covers five minutes. It is a terminal/file view,
not a network-exposed dashboard.

Read it from the deployment directory:

```bash
watch -n 5 cat results/serving-monitor/latest.md
```

## Measurements and interpretation

Reuse `vllm:request_time_per_output_token_seconds` as the headline outcome. In
this vLLM version it is `(last_token_time - first_token_time) /
(generated_tokens - 1)` and is observed when a request completes. It counts real
output tokens, including speculative tokens, and excludes the initial prefill
token. It includes preemptions during decode. Histogram deltas give the average
of per-request means for requests completing in that interval, not a
per-token-weighted mean. A request spanning a configuration change is mixed
exposure and must not be assigned entirely to its completion-time configuration.

Reuse inter-token latency to assess output cadence and tails. MTP can emit
multiple tokens per output, so this is not equivalent to per-token latency.
Histogram percentile bucket bounds are shown as bounds, not invented exact
percentiles.

Prometheus supplies existing outcome metrics and workload counters. Existing
scheduler accounting logs supply context lengths without collecting request
contents or identifiers. Small uprobes on the already-loaded, hash-verified PLE
libraries record one step record and one preceding-step GPU timing record per
rank. There are no added CUDA kernels or library changes. Probes have overhead;
its magnitude has not been established, so keep the same instrumentation across
comparisons. No Nsight profiler is started.

PLE data includes native read time, hash-to-GPU-ready gate, GPU wait-loop time,
and CPU publication-to-next-publication step interval. The gate includes
intervening GPU work, is not pure I/O time, and does not establish GPU L2
residency. Consecutive-step intervals are only grouped when both batches have
the same request/token/row shape and deferred graph execution.

Groups separate worker epoch, rank, active batch size, 4K mean-context bins,
recent prefill activity, and 10-percentage-point speculative acceptance bins.
Context and acceptance are recent aggregate observations, not exact per-step
values. Stale accounting, mismatched resident/batch request counts, and shape
transitions are excluded. Sample counts and occupied ten-second time blocks
are shown. Fewer than100 samples or five blocks is explicitly sparse. Passing
those thresholds means descriptive coverage, not statistical proof.

This collector does NOT automatically rank implementations. A valid future
comparison needs overlapping workload groups, repeated time blocks, a recorded
configuration-change boundary, and separation of requests spanning that boundary.
The API outcomes are common to the original and replacement implementations.
PLE component probes currently support the validated reader17/hash12/wait12 ABI;
an unknown implementation disables component probing rather than guessing
memory offsets. API collection continues. A worker exit causes probes to
reattach to the next compatible worker; metrics counter resets start a new delta.

## Operation

```bash
systemctl --user status ple-serving-monitor
systemctl --user stop ple-serving-monitor
```

Shutdown sends SIGINT only to this monitor's two named bpftrace invocations.
The transient unit does not persist across host reboots. Start again with:

```bash
systemd-run --user --unit=ple-serving-monitor \
  --property=WorkingDirectory=/home/jon/dual-spark-inference-kv-paging \
  --property=Restart=on-failure --property=RestartSec=10 \
  --property=TimeoutStopSec=25 \
  /home/jon/dual-spark-inference-kv-paging/.venv/bin/python \
  /home/jon/dual-spark-inference-kv-paging/experiments/serving-monitor/monitor.py
```

Startup validation covered empty snapshots and absent first metric intervals.
Live acceptance confirmed both ranks collecting, valid matching sequence
numbers, no native/GPU errors, and empty probe stderr. Initial collector startup
failures were fixed; their probes were removed before the accepted launch.
No inference service restart occurred.
