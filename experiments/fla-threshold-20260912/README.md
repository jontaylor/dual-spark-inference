# GB10 FLA shared-memory threshold test — 2026-09-12

## Result and current state

The 99 KiB threshold is enabled on both running ranks through a hash-verified,
read-only runtime override. No model weights, sampling settings, or other kernel
fixes were changed. The original image is unchanged. This was a small smoke and
performance experiment, not a broad quality qualification.

The installed baseline required 102400 bytes, but GB10 reports 101376. Its default
check returned false. The patch changes only `Backend.DEFAULT` to 101376. The
server log confirms Triton/FLA GDN prefill, with a separate CUDA decode path.
The output-kernel autotuning candidates change from [32,64] to [64,128]; both
include 64, so enabling the gate does not guarantee a speedup.

## Measurements

Actual model: nvidia/Qwen3.8-Flash-Next-NVFP4, revision
fc694b54fb0174e0913e6adf86691ef85a4ead47. Two GB10s, TP2+EP, MTP3, BF16 KV,
existing paging stack. The existing ten-trial planner workload remained running
with its sampling, cadence and retries unchanged. It retried through restart.

Paired full-FLA-operation test used the model's per-rank dimensions: 8 key heads,
24 value heads, K=V=128, BF16 inputs with normalized queries/keys. Each shape had
12 alternating A/B timing rounds, three calls per timing interval, after tuning.
Patched modules were imported only in the probe process; the initial probe did
not mutate the serving process. Both variants used identical seeded inputs.

| Input lengths | Baseline median ms | Patched median ms | Baseline / patched |
|---|---:|---:|---:|
| 128 | 0.0654 | 0.0643 | 1.017 |
| 2048 | 1.8102 | 1.8202 | 0.994 |
| 8192 | 9.3125 | 9.3956 | 0.991 |
| 511 + 1025 + 513 | 1.8870 | 1.8694 | 1.009 |

No clear isolated operation speedup: differences are within about 2%, with
substantial interference from the background GPU workload. Final recurrent states
were exactly equal for all four cases. Outputs were finite; relative L2
differences were 1.20e-5 to 1.71e-5, maximum absolute difference 0.00024414.
They were not bit-identical. These checks do not establish broad model quality.

Cold-prefix API probes used a unique initial identifier (zero cached prompt
tokens), embedded-code retrieval plus arithmetic, temperature zero, thinking off,
and a 48-token output ceiling. All returned the exact expected answer. Actual
prompt counts are approximately 2048 and 8192, recorded in the JSON artifacts.

| Prompt target | Baseline TTFT seconds (3 samples) | Patched TTFT seconds (3 samples) | Median change |
|---|---|---|---:|
| 2048 | 0.869, 0.945, 0.958 | 0.875, 1.132, 0.823 | -7.5% |
| 8192 | 3.462, 3.157, 6.542 | 3.325, 3.222, 2.886 | -6.9% |

The six measured baseline and six measured patched answers all passed. Six
additional patched warmup answers also passed. Restart/cache rebuild and warmup
were excluded. `baseline.json` overlaps the isolated GPU test and is excluded
from performance comparisons; use `baseline-clean.json` instead.

The approximately 7% API improvement is only a hint: sample size is small,
background request mix changes, and baseline/patched runs have different cache
histories after the restart. There was no A-B-A restart control. Do not claim a
proven 7% improvement, a decode speedup, or the upstream headline gain. The
isolated operation results do not independently support a substantial gain.

## Files and reproduction

- `utils.baseline.py` / `utils.patched.py`: exact sources.
- `kernel_ab.py`, `kernel-ab.log`, `kernel-results.json`: paired operation test.
- `api_probe.py`, `baseline-clean.json`, `patched-clean.json`: API probes/results.
- `patched-warmup.json`: startup/recovery samples, excluded from timings.
- `config-before.json`: original local runtime config on each respective host.
- `set_arm.py`: adds/removes just this runtime override and its hash.
- `experiment.json`: setup metadata.

For another API smoke run on spark-1:

```bash
python3 /home/jon/dual-spark-inference-kv-paging/experiments/fla-threshold-20260912/api_probe.py followup
```

## Rollback

Run once on spark-1; these commands remove only this override, preserving other
config fields. A restart drops active requests; existing clients retry. Rebuilding
long prefixes takes additional time after the API becomes healthy.

```bash
python3 /home/jon/dual-spark-inference-kv-paging/experiments/fla-threshold-20260912/set_arm.py baseline
ssh 192.168.100.11 'python3 /home/jon/dual-spark-inference-kv-paging/experiments/fla-threshold-20260912/set_arm.py baseline'
sudo systemctl restart qwen38-next-qwen-fp8.service
```
