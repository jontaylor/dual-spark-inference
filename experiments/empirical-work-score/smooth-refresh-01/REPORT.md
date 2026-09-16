# Smooth empirical work scoring

Fits use raw 30-second observations, not bucket maxima. Reference is the 90% conditional quantile of positive throughput. Runs have equal total fitting weight. Zero-rate windows are excluded from capacity fitting but earn zero credit for that operation during scoring. All archived reference windows are used, including after tasks finish.

Candidate selection: leave one complete run out at a time; evaluate log-throughput quantile loss, averaged equally across held-out runs. Select the simplest candidate within one standard error of the lowest loss. This assesses throughput fit, not prediction of task success. The quantile is an adjustable policy choice, not physical peak capacity.

| Operation | Selected function | Lowest CV loss candidate | Held-out observations below reference |
|---|---|---|---:|
| decode | log-polynomial-1 | log-polynomial-2 | 89.6% |
| prefill | log-polynomial-1 | log-polynomial-1 | 89.6% |

## Functions

Use `WorkFunctions("path/to/functions.json")` from `fit_functions.py`. It exposes `decode_per_second(context)`, `prefill_per_second(context)`, `decode_units(context, rate)` and `prefill_units(context, rate)`. Context is tokens; rate is cluster tokens/s. Units = observed rate / fitted reference; total = decode units + prefill units. Values above one are allowed.

Exact representation: z = (log(1 + context/1000) − log_min)/(log_max − log_min); reference = exp(B(z) · coefficients). B is either a polynomial basis or cubic B-spline basis. Coefficients and knots are in functions.json. Context is clamped to each operation’s observed domain; no unsupported extrapolation.

## Minutes 30–35

| Rank | Run | Decode units/s | Prefill units/s | Combined units/s | Scored seconds |
|---:|---|---:|---:|---:|---:|
| 1 | 18-s32-b8192-t2048 | 0.690 | 0.893 | 1.582 | 300 |
| 2 | 08-s24-b8192-t2048 | 0.841 | 0.644 | 1.485 | 300 |
| 3 | 06-s24-b32768-t1024 | 0.661 | 0.797 | 1.458 | 300 |
| 4 | 05-s24-b16384-t1024 | 0.781 | 0.620 | 1.402 | 300 |
| 5 | 04-s24-b4096-t1024 | 0.846 | 0.445 | 1.291 | 300 |
| 6 | 03-s32-b8192-t1024 | 0.948 | 0.342 | 1.290 | 300 |
| 7 | 01-s24-b8192-t1024 | 0.898 | 0.276 | 1.174 | 300 |
| 8 | 12-s32-b16384-t2048 | 0.625 | 0.479 | 1.103 | 300 |
| 9 | 07-s24-b8192-t512 | 0.681 | 0.408 | 1.089 | 300 |
| 10 | 19-s32-b4096-t2048 | 0.745 | 0.159 | 0.904 | 270 |
| 11 | 09-s8-b4096-t2048 | 0.519 | 0.262 | 0.781 | 300 |
| 12 | 02-s8-b8192-t1024 | 0.554 | 0.188 | 0.742 | 300 |

## First 35 minutes

| Run | Combined units/s | Scored seconds |
|---|---:|---:|
| 01-s24-b8192-t1024 | 1.263 | 2070 |
| 02-s8-b8192-t1024 | 0.832 | 2070 |
| 03-s32-b8192-t1024 | 1.455 | 2070 |
| 04-s24-b4096-t1024 | 1.309 | 2070 |
| 05-s24-b16384-t1024 | 1.406 | 2070 |
| 06-s24-b32768-t1024 | 1.353 | 2070 |
| 07-s24-b8192-t512 | 1.156 | 2070 |
| 08-s24-b8192-t2048 | 1.254 | 2070 |
| 09-s8-b4096-t2048 | 0.690 | 2070 |
| 12-s32-b16384-t2048 | 1.390 | 2070 |
| 18-s32-b8192-t2048 | 1.286 | 2070 |
| 19-s32-b4096-t2048 | 1.188 | 2040 |

## Sensitivity

0.85 reference: 18 → 06 → 08 → 05 → 04 → 03 → 01 → 12 → 07 → 19 → 09 → 02

0.95 reference: 18 → 08 → 06 → 05 → 03 → 04 → 01 → 12 → 07 → 19 → 09 → 02


## Candidate validation

| Operation | Candidate | CV loss | Standard error |
|---|---|---:|---:|
| decode | log-polynomial-1 | 0.0484 | 0.0034 |
| decode | log-polynomial-2 | 0.0459 | 0.0036 |
| decode | cubic-spline-0-knots | 0.0461 | 0.0036 |
| decode | cubic-spline-2-knots | 0.0469 | 0.0036 |
| decode | cubic-spline-4-knots | 0.0469 | 0.0036 |
| prefill | log-polynomial-1 | 0.1497 | 0.0080 |
| prefill | log-polynomial-2 | 0.1499 | 0.0081 |
| prefill | cubic-spline-0-knots | 0.1498 | 0.0080 |
| prefill | cubic-spline-2-knots | 0.1500 | 0.0079 |
| prefill | cubic-spline-4-knots | 0.1501 | 0.0081 |

![Smooth reference curves](curves.png)

Limitations inherited from source data: context is estimated from overlapping HTTP requests, including queue time and uniform output growth, not measured GPU context. Both operations share that context estimate. Calls missing final usage may be absent. Run 9 includes a profiler overrun. Initial windows lacking context coverage are unscored. Negative prefill rates smaller than 1e-6 tok/s in magnitude are floating-point subtraction noise and are clamped to zero. Sparse high-context coverage, batch occupancy and operation mix affect empirical fits. The 85–95% band shows reference-choice sensitivity, not statistical confidence. No inference services or campaign files were modified.

Audit: functions.json includes exact coefficients, validation folds and source hash; reference-curves.csv, scored-windows.csv and scores.csv provide numerical outputs.

## Refresh-specific coverage note

This refresh includes completed runs 1–9, 12, 18 and 19, fitted and rescored together. Its numerical scale differs from smooth-v2 because the reference cohort changed. Failed, skipped and live attempts are excluded. Run 12 first recorded a completed task at 33m01.9s after launch (task duration 32m56.3s), so the last roughly two minutes of its first-35-minute interval no longer satisfy the earlier assumption that no whole task has finished. Runs 18 and 19 first observed completions after minute 35. All 1,557 exported scored windows were independently recomputed successfully.
