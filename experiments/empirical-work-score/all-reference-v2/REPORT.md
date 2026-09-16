# Empirical work score

Rates are measured; context is estimated from overlapping HTTP requests, including queued requests. Output growth is assumed uniform across request lifetime. Requests without final usage/timing are excluded, so still-running calls at archival cutoff may be absent. No actual GPU context or token emission timing was recorded.

Reference: maximum observed 30-second rate in each 1000-token context bucket, pooled over completed runs (reference horizon: all available minutes). Decode and prefill use the same mean context estimate. Peaks may come from different windows. Combined score = decode/peak_decode + uncached_prefill/peak_prefill. This is an empirical index, not physical GPU utilisation; values above 1 are allowed.

Only windows with at least 95% reconstructed-context coverage and nonnegative counter deltas are scored. Missing or zero denominators give an unknown score. Scores average only covered windows; inspect valid_seconds before comparing. Sparse buckets and winners setting their own reference remain visible in peaks.csv. Reference changes when the source cohort changes. Lines connect occupied bucket centres for display; scoring uses the bucket maxima directly, without interpolation.

| Run | Interval | Scored seconds | Mean units/s |
|---|---|---:|---:|
| 01-s24-b8192-t1024 | 0-35m | 2070 | 1.0988 |
| 01-s24-b8192-t1024 | 30-35m | 300 | 0.9515 |
| 02-s8-b8192-t1024 | 0-35m | 2070 | 0.7197 |
| 02-s8-b8192-t1024 | 30-35m | 300 | 0.6719 |
| 03-s32-b8192-t1024 | 0-35m | 2070 | 1.2372 |
| 03-s32-b8192-t1024 | 30-35m | 300 | 1.0769 |
| 04-s24-b4096-t1024 | 0-35m | 2070 | 1.1265 |
| 04-s24-b4096-t1024 | 30-35m | 300 | 1.2554 |
| 05-s24-b16384-t1024 | 0-35m | 2070 | 1.1930 |
| 05-s24-b16384-t1024 | 30-35m | 300 | 1.0124 |
| 06-s24-b32768-t1024 | 0-35m | 2070 | 1.1533 |
| 06-s24-b32768-t1024 | 30-35m | 300 | 1.1684 |
| 07-s24-b8192-t512 | 0-35m | 2070 | 0.9505 |
| 07-s24-b8192-t512 | 30-35m | 300 | 0.7380 |
| 08-s24-b8192-t2048 | 0-35m | 2040 | 1.1134 |
| 08-s24-b8192-t2048 | 30-35m | 300 | 1.0950 |
| 09-s8-b4096-t2048 | 0-35m | 2070 | 0.6629 |
| 09-s8-b4096-t2048 | 30-35m | 300 | 0.6081 |

![Empirical curves](curves.png)

Raw data: samples.csv, peaks.csv, scores.csv. Source paths, request exclusions and profiler caveats: manifest.json. No live services or campaign files were modified.
