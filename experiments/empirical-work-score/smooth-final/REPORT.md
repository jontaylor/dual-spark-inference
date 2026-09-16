# Smooth empirical work scoring

Fits use raw 30-second observations, not bucket maxima. Reference is the 90% conditional quantile of positive throughput. Runs have equal total fitting weight. Zero-rate windows are excluded from capacity fitting but earn zero credit for that operation during scoring. All archived reference windows are used, including after tasks finish.

Candidate selection: leave one complete run out at a time; evaluate log-throughput quantile loss, averaged equally across held-out runs. Select the simplest candidate within one standard error of the lowest loss. This assesses throughput fit, not prediction of task success. The quantile is an adjustable policy choice, not physical peak capacity.

| Operation | Selected function | Lowest CV loss candidate | Held-out observations below reference |
|---|---|---|---:|
| decode | log-polynomial-1 | log-polynomial-2 | 89.4% |
| prefill | log-polynomial-1 | cubic-spline-2-knots | 89.8% |

## Functions

Use `WorkFunctions("path/to/functions.json")` from `fit_functions.py`. It exposes `decode_per_second(context)`, `prefill_per_second(context)`, `decode_units(context, rate)` and `prefill_units(context, rate)`. Context is tokens; rate is cluster tokens/s. Units = observed rate / fitted reference; total = decode units + prefill units. Values above one are allowed.

Exact representation: z = (log(1 + context/1000) − log_min)/(log_max − log_min); reference = exp(B(z) · coefficients). B is either a polynomial basis or cubic B-spline basis. Coefficients and knots are in functions.json. Context is clamped to each operation’s observed domain; no unsupported extrapolation.

## Minutes 30–35

| Rank | Run | Decode units/s | Prefill units/s | Combined units/s | Scored seconds |
|---:|---|---:|---:|---:|---:|
| 1 | 18-s32-b8192-t2048 | 0.696 | 0.863 | 1.559 | 300 |
| 2 | 08-s24-b8192-t2048 | 0.849 | 0.622 | 1.472 | 300 |
| 3 | 06-s24-b32768-t1024 | 0.670 | 0.768 | 1.437 | 300 |
| 4 | 05-s24-b16384-t1024 | 0.790 | 0.599 | 1.389 | 300 |
| 5 | 03-s32-b8192-t1024 | 0.959 | 0.330 | 1.289 | 300 |
| 6 | 04-s24-b4096-t1024 | 0.858 | 0.428 | 1.286 | 300 |
| 7 | 36-s32-b16384-t1024 | 0.778 | 0.434 | 1.212 | 300 |
| 8 | 01-s24-b8192-t1024 | 0.908 | 0.267 | 1.175 | 300 |
| 9 | 12-s32-b16384-t2048 | 0.628 | 0.465 | 1.093 | 300 |
| 10 | 07-s24-b8192-t512 | 0.685 | 0.396 | 1.081 | 300 |
| 11 | 35-s32-b4096-t1024 | 0.610 | 0.336 | 0.946 | 300 |
| 12 | 19-s32-b4096-t2048 | 0.746 | 0.155 | 0.901 | 270 |
| 13 | 09-s8-b4096-t2048 | 0.521 | 0.255 | 0.775 | 300 |
| 14 | 02-s8-b8192-t1024 | 0.561 | 0.181 | 0.743 | 300 |

## First 35 minutes

| Run | Combined units/s | Scored seconds |
|---|---:|---:|
| 01-s24-b8192-t1024 | 1.255 | 2070 |
| 02-s8-b8192-t1024 | 0.827 | 2070 |
| 03-s32-b8192-t1024 | 1.445 | 2070 |
| 04-s24-b4096-t1024 | 1.300 | 2070 |
| 05-s24-b16384-t1024 | 1.397 | 2070 |
| 06-s24-b32768-t1024 | 1.340 | 2070 |
| 07-s24-b8192-t512 | 1.148 | 2070 |
| 08-s24-b8192-t2048 | 1.244 | 2070 |
| 09-s8-b4096-t2048 | 0.684 | 2070 |
| 12-s32-b16384-t2048 | 1.378 | 2070 |
| 18-s32-b8192-t2048 | 1.276 | 2070 |
| 19-s32-b4096-t2048 | 1.179 | 2040 |
| 35-s32-b4096-t1024 | 1.281 | 2070 |
| 36-s32-b16384-t1024 | 1.309 | 2070 |

## Sensitivity

0.85 reference: 18 → 06 → 08 → 05 → 04 → 03 → 36 → 01 → 12 → 07 → 35 → 19 → 09 → 02

0.95 reference: 18 → 08 → 06 → 05 → 03 → 04 → 36 → 01 → 12 → 07 → 35 → 19 → 09 → 02


## Candidate validation

| Operation | Candidate | CV loss | Standard error |
|---|---|---:|---:|
| decode | log-polynomial-1 | 0.0498 | 0.0032 |
| decode | log-polynomial-2 | 0.0475 | 0.0033 |
| decode | cubic-spline-0-knots | 0.0476 | 0.0033 |
| decode | cubic-spline-2-knots | 0.0481 | 0.0033 |
| decode | cubic-spline-4-knots | 0.0484 | 0.0033 |
| prefill | log-polynomial-1 | 0.1562 | 0.0090 |
| prefill | log-polynomial-2 | 0.1564 | 0.0090 |
| prefill | cubic-spline-0-knots | 0.1559 | 0.0087 |
| prefill | cubic-spline-2-knots | 0.1559 | 0.0087 |
| prefill | cubic-spline-4-knots | 0.1562 | 0.0089 |

![Smooth reference curves](curves.png)

Limitations inherited from source data: context is estimated from overlapping HTTP requests, including queue time and uniform output growth, not measured GPU context. Both operations share that context estimate. Calls missing final usage may be absent. Run 9 includes a profiler overrun. Initial windows lacking context coverage are unscored. Negative prefill rates smaller than 1e-6 tok/s in magnitude are floating-point subtraction noise and are clamped to zero. Sparse high-context coverage, batch occupancy and operation mix affect empirical fits. The 85–95% band shows reference-choice sensitivity, not statistical confidence. No inference services or campaign files were modified.

Audit: functions.json includes exact coefficients, validation folds and source hash; reference-curves.csv, scored-windows.csv and scores.csv provide numerical outputs.
