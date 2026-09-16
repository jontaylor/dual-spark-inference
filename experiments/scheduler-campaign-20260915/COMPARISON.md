# Scheduler campaign comparison

8/8 measured configurations available. Each arm uses a fresh campaign and a600-second window; counter rates span the300 samples (approximately598seconds).

| Sequences | Budget | Threshold | Gen tok/s | Gen tok/s final5m | Mean TTFT s | Prompt reuse % | Completed requests | GPU power W combined |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 16384 | 2048 | 321.51 | 337.78 | 3.77 | 89.39 | 260 | 93.62 |
| 32 | 16384 | 512 | 276.60 | 320.11 | 6.57 | 88.14 | 256 | 95.68 |
| 32 | 4096 | 512 | 271.82 | 281.14 | 3.49 | 93.68 | 275 | 90.71 |
| 32 | 4096 | 2048 | 327.16 | 355.59 | 3.29 | 89.50 | 240 | 93.01 |
| 24 | 4096 | 2048 | 311.22 | 337.15 | 3.27 | 91.24 | 288 | 93.82 |
| 24 | 4096 | 512 | 250.71 | 283.62 | 6.31 | 89.95 | 208 | 91.90 |
| 24 | 16384 | 512 | 257.45 | 250.81 | 18.69 | 85.11 | 280 | 97.48 |
| 24 | 16384 | 2048 | 280.48 | 316.75 | 4.24 | 88.86 | 291 | 96.00 |

## Matched configuration pairs

Each row changes only the named configuration field. These are observations from single runs, not estimates with statistical confidence. Model/tool trajectories, prompt work and cache reuse can diverge even with the same frozen workload.

| Change | Other settings | Generation throughput change | Mean TTFT change |
|---|---|---:|---:|
| Sequences 24 → 32 | budget=4096, threshold=2048 | +5.12% | +0.02s |
| Sequences 24 → 32 | budget=4096, threshold=512 | +8.42% | -2.82s |
| Sequences 24 → 32 | budget=16384, threshold=512 | +7.44% | -12.12s |
| Sequences 24 → 32 | budget=16384, threshold=2048 | +14.63% | -0.48s |
| Budget 4096 → 16384 | seq=32, threshold=512 | +1.76% | +3.08s |
| Budget 4096 → 16384 | seq=32, threshold=2048 | -1.72% | +0.47s |
| Budget 4096 → 16384 | seq=24, threshold=2048 | -9.88% | +0.97s |
| Budget 4096 → 16384 | seq=24, threshold=512 | +2.69% | +12.38s |
| Threshold 512 → 2048 | seq=32, budget=16384 | +16.24% | -2.80s |
| Threshold 512 → 2048 | seq=32, budget=4096 | +20.36% | -0.20s |
| Threshold 512 → 2048 | seq=24, budget=4096 | +24.14% | -3.04s |
| Threshold 512 → 2048 | seq=24, budget=16384 | +8.94% | -14.44s |

## Prompt work and reported transfers

| Arm | Local compute tokens | Local-cache tokens | External-path tokens | Reported GPU→CPU bytes | Reported CPU→GPU bytes |
|---|---:|---:|---:|---:|---:|
| 01-s32-b16384-t2048 | 349032 | 633600 | 2305750 | 0 | 0 |
| 02-s32-b16384-t512 | 497102 | 1461120 | 2232578 | 0 | 0 |
| 03-s32-b4096-t512 | 278032 | 0 | 4120032 | 0 | 0 |
| 04-s32-b4096-t2048 | 274935 | 238080 | 2106299 | 0 | 0 |
| 05-s24-b4096-t2048 | 307700 | 0 | 3205294 | 0 | 0 |
| 06-s24-b4096-t512 | 413736 | 1520640 | 2181096 | 0 | 0 |
| 07-s24-b16384-t512 | 470115 | 1142400 | 1544220 | 0 | 0 |
| 08-s24-b16384-t2048 | 484614 | 1655040 | 2210797 | 0 | 0 |

External-path reuse includes the custom GPU-resident completion cache. These counters must not be interpreted as physical disk I/O. Host diskstats and raw connector logs are retained for further attribution.

## Interpretation limits

- No repeat runs or ninth drift-control run; the eight combinations were tested in the agreed order. Filesystem PLE cache was not flushed.
- All serving settings other than the three matrix fields were preserved. The existing fair-prefill and cache/recurrent alignment policies can reduce actual chunks below the threshold. Sampled chunk histograms are in results.json; they are not a complete per-step trace.
- Readiness/campaign reset and collection audit is in audit.json. Sampling errors in nonblocking Python profiles are preserved in the profile logs; these are not GPU kernel traces.
- Generated tokens include partial responses at cutoff. Mean latency includes observations recorded during the window, not an identical request cohort. Campaign run states and calls/tool events are separately recorded. No task-quality or deterministic-output conclusion is established.
- Arm6 had one disk-reserve preflight failure before measurement; it was recovered without changing serving settings. See RECOVERY.md. Full historical archives were relocated to the worker with hash verification; paths are in archive-relocations.json.
- The final matrix configuration stays running; the campaign does not automatically deploy whichever configuration has the largest observed throughput.
