# Scheduler campaign results

All eight ten-minute arms are complete. Throughputs use counter deltas and include unfinished responses. Prompt throughput includes cached tokens. No full determinism claim.

| Arm | Generation tok/s · full | Generation tok/s · final 5m | Prompt tok/s · full | Same campaign throughout |
|---|---:|---:|---:|---|
| 01-s32-b16384-t2048 | 321.51 | 337.78 | 5498.37 | True |
| 02-s32-b16384-t512 | 276.60 | 320.11 | 7007.27 | True |
| 03-s32-b4096-t512 | 271.82 | 281.14 | 7353.79 | True |
| 04-s32-b4096-t2048 | 327.16 | 355.59 | 4379.63 | True |
| 05-s24-b4096-t2048 | 311.22 | 337.15 | 5873.91 | True |
| 06-s24-b4096-t512 | 250.71 | 283.62 | 6881.35 | True |
| 07-s24-b16384-t512 | 257.45 | 250.81 | 5278.18 | True |
| 08-s24-b16384-t2048 | 280.48 | 316.75 | 7274.19 | True |

| Arm | Cached prompt % | Mean TTFT · seconds | Completed requests | Running / waiting at cutoff |
|---|---:|---:|---:|---:|
| 01-s32-b16384-t2048 | 89.39 | 3.77 | 260 | 24 / 0 |
| 02-s32-b16384-t512 | 88.14 | 6.57 | 256 | 24 / 0 |
| 03-s32-b4096-t512 | 93.68 | 3.49 | 275 | 11 / 2 |
| 04-s32-b4096-t2048 | 89.50 | 3.29 | 240 | 24 / 0 |
| 05-s24-b4096-t2048 | 91.24 | 3.27 | 288 | 24 / 0 |
| 06-s24-b4096-t512 | 89.95 | 6.31 | 208 | 16 / 0 |
| 07-s24-b16384-t512 | 85.11 | 18.69 | 280 | 24 / 11 |
| 08-s24-b16384-t2048 | 88.86 | 4.24 | 291 | 24 / 0 |

Campaign run statuses and recorded call/tool counts are preserved in results.json. Completed inference requests are not completed benchmark tasks.

Latency averages cover observations recorded during the window, not a fixed cohort. External-cache source labels include the custom GPU-resident completion restore path and must not be equated with disk traffic.

Raw counters, histogram sums/counts, sampled scheduler allocations, GPU summaries and cutoff request counts are in results.json. Campaign artifacts are in each arm’s campaign-evidence.tar.gz. Five-second GPU samples cannot measure every brief stall. Python stack samples are not GPU kernel traces.
