# Completion campaign results

Scope revised by user: skip all s8, s24 and t512. Historical completed results below are retained; skipped partial runs are not measurements. See scope-revision.json.
6 / 6 currently requested configurations complete; 14 total historical measurements retained. Primary endpoint: eighth normally completed task; this is not proof of quality-test success. Crossing intervals reflect polling.

| Configuration | Time to eighth from launch (s) | Polling interval (s) | Generated tok/s | Uncached prompt tok/s | Prompt reuse % |
|---|---:|---:|---:|---:|---:|
| 01-s24-b8192-t1024 | 4017.0 | 4011.0–4017.0 | 246.96 | 347.88 | 95.54 |
| 02-s8-b8192-t1024 | 4340.0 | 4334.2–4340.0 | 167.03 | 225.89 | 97.20 |
| 03-s32-b8192-t1024 | 4414.6 | 4408.6–4414.6 | 256.53 | 339.13 | 95.58 |
| 04-s24-b4096-t1024 | 4461.4 | 4455.4–4461.4 | 226.60 | 303.13 | 96.13 |
| 05-s24-b16384-t1024 | 3278.1 | 3272.1–3278.1 | 262.02 | 333.37 | 96.63 |
| 06-s24-b32768-t1024 | 3344.0 | 3338.1–3344.0 | 238.07 | 361.85 | 96.17 |
| 07-s24-b8192-t512 | 5376.8 | 5370.8–5376.8 | 212.25 | 298.70 | 95.39 |
| 08-s24-b8192-t2048 | 3034.6 | 3028.5–3034.6 | 267.59 | 270.55 | 97.20 |
| 09-s8-b4096-t2048 | 5850.8 | 5844.8–5850.8 | 171.99 | 176.38 | 96.88 |
| 12-s32-b16384-t2048 | 2828.1 | 2822.0–2828.1 | 247.31 | 391.13 | 96.02 |
| 18-s32-b8192-t2048 | 3326.4 | 3320.3–3326.4 | 252.21 | 297.45 | 97.06 |
| 19-s32-b4096-t2048 | 2996.4 | 2990.4–2996.4 | 268.62 | 285.96 | 96.15 |
| 35-s32-b4096-t1024 | 4015.8 | 4009.8–4015.8 | 202.29 | 375.18 | 95.64 |
| 36-s32-b16384-t1024 | 4132.3 | 4126.3–4132.3 | 247.14 | 286.04 | 96.30 |

Full counter and label deltas, task identities and quality evidence are in results.json. See [QUALITY.md](QUALITY.md) for benchmark, original CLI and supplementary verification of cutoff finishers. Reuse includes GPU-resident completion restores; external labels are not equivalent to disk reads. Different completed task subsets and trajectories still limit causal comparisons.

Final run 36-s32-b16384-t1024: 16 tasks completed in 5128.4s from launch; full-run generation 214.93 tok/s. Comparison table uses the first-eight window; full counters are separate in results.json.

## Measurement caveats

- **09-s8-b4096-t2048**: Requested 20s capture still running after 156s at 97.6 percent CPU; API health 200. Overrun capture; exclude from standard 20s profile comparisons. Measurement includes profiler overhead. See [09-s8-b4096-t2048/profile-120-worker-intervention.json](09-s8-b4096-t2048/profile-120-worker-intervention.json).
- **36-s32-b16384-t1024**: Worker profiler reported1.15s behind in sampling at120-second capture. Profile is valid JSON with2820samples and29sampling errors; sampling timing and attribution may be inaccurate. No profiler intervention or workload change was made. See [36-s32-b16384-t1024/profile-120-worker-intervention.json](36-s32-b16384-t1024/profile-120-worker-intervention.json).

## Failed attempts — not completed measurements

- **13-s32-b32768-t2048**: supervisor_memory_pressure_shutdown. rank0 MemAvailable below 8 GiB for five consecutive samples; final 5.16 GiB, swap used 16.0 GiB. See [13-s32-b32768-t2048/failed.json](13-s32-b32768-t2048/failed.json).
- **34-s32-b32768-t1024**: supervisor_memory_pressure_shutdown. rank0 MemAvailable below 8 GiB for five consecutive samples; final 7.96 GiB, free 4.88 GiB, swap used 15.835 GiB at 1789501172.7062294. See [34-s32-b32768-t1024/failed.json](34-s32-b32768-t1024/failed.json).
