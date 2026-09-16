# Final scheduler work analysis

The campaign has ended under its revised scope. There are 14 completed historical measurements; the final six requested configurations have measurements. Two s32/b32768 attempts ended under memory pressure and were not retried. This report performs offline analysis only; no services or controller state were changed.

**Conclusion:** run 3 (s32/b8192/t1024) leads the context-adjusted first-35-minute work score. The predicted s32/b16384/t1024 improvement did not materialise. This is the strongest measured candidate under the chosen score, not a statistically established optimum.

All 14 runs were refitted and rescored against a common 90th-percentile empirical reference. Scores are not directly comparable in absolute scale to previous report versions. The reference uses all archived windows; run 36 alone continued until all 16 tasks finished. Its longer tail is included in reference fitting, with equal total fitting weight per run.

| Work rank | Run / settings | Decode units/s | Prefill units/s | Combined | Scored seconds | Time to 8 | Supplementary pass/fail/unknown, first 8 |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | 03-s32-b8192-t1024 | 0.912 | 0.533 | 1.445 | 2070 | 73m 34.6s | 6/2/0 |
| 2 | 05-s24-b16384-t1024 | 0.902 | 0.495 | 1.397 | 2070 | 54m 38.1s | 5/3/0 |
| 3 | 12-s32-b16384-t2048 | 0.810 | 0.568 | 1.378 | 2070 | 47m 08.1s | 1/7/0 |
| 4 | 06-s24-b32768-t1024 | 0.790 | 0.550 | 1.340 | 2070 | 55m 44.0s | 6/2/0 |
| 5 | 36-s32-b16384-t1024 | 0.883 | 0.426 | 1.309 | 2070 | 68m 52.3s | 1/7/0 |
| 6 | 04-s24-b4096-t1024 | 0.808 | 0.492 | 1.300 | 2070 | 74m 21.4s | 4/4/0 |
| 7 | 35-s32-b4096-t1024 | 0.762 | 0.519 | 1.281 | 2070 | 66m 55.8s | 7/1/0 |
| 8 | 18-s32-b8192-t2048 | 0.837 | 0.439 | 1.276 | 2070 | 55m 26.4s | 6/2/0 |
| 9 | 01-s24-b8192-t1024 | 0.849 | 0.406 | 1.255 | 2070 | 66m 57.0s | 3/5/0 |
| 10 | 08-s24-b8192-t2048 | 0.869 | 0.376 | 1.244 | 2070 | 50m 34.6s | 1/7/0 |
| 11 | 19-s32-b4096-t2048 | 0.847 | 0.332 | 1.179 | 2040 | 49m 56.4s | 2/6/0 |
| 12 | 07-s24-b8192-t512 | 0.749 | 0.398 | 1.148 | 2070 | 89m 36.8s | 6/2/0 |
| 13 | 02-s8-b8192-t1024 | 0.525 | 0.301 | 0.827 | 2070 | 72m 20.0s | 4/4/0 |
| 14 | 09-s8-b4096-t2048 | 0.472 | 0.212 | 0.684 | 2070 | 97m 30.8s | 7/1/0 |

## Comparisons changing one setting

These are observed differences in single runs, not causal parameter estimates. Changing one setting still changed the model trajectory.

| Change | Other settings | Score change |
|---|---|---:|
| sequences 8 → 32 | b=4096, t=2048 | +72.4% |
| sequences 8 → 24 | b=8192, t=1024 | +51.8% |
| sequences 8 → 32 | b=8192, t=1024 | +74.8% |
| sequences 24 → 32 | b=4096, t=1024 | -1.5% |
| sequences 24 → 32 | b=8192, t=1024 | +15.1% |
| sequences 24 → 32 | b=8192, t=2048 | +2.6% |
| sequences 24 → 32 | b=16384, t=1024 | -6.3% |
| budget 4096 → 8192 | s=24, t=1024 | -3.4% |
| budget 4096 → 16384 | s=24, t=1024 | +7.5% |
| budget 4096 → 32768 | s=24, t=1024 | +3.1% |
| budget 8192 → 16384 | s=24, t=1024 | +11.3% |
| budget 8192 → 32768 | s=24, t=1024 | +6.8% |
| budget 16384 → 32768 | s=24, t=1024 | -4.1% |
| budget 4096 → 8192 | s=32, t=1024 | +12.8% |
| budget 4096 → 16384 | s=32, t=1024 | +2.2% |
| budget 4096 → 8192 | s=32, t=2048 | +8.2% |
| budget 4096 → 16384 | s=32, t=2048 | +16.9% |
| budget 8192 → 16384 | s=32, t=1024 | -9.4% |
| budget 8192 → 16384 | s=32, t=2048 | +8.0% |
| threshold 512 → 1024 | s=24, b=8192 | +9.4% |
| threshold 512 → 2048 | s=24, b=8192 | +8.4% |
| threshold 1024 → 2048 | s=24, b=8192 | -0.9% |
| threshold 1024 → 2048 | s=32, b=4096 | -7.9% |
| threshold 1024 → 2048 | s=32, b=8192 | -11.7% |
| threshold 1024 → 2048 | s=32, b=16384 | +5.3% |

## Robustness and coverage

The first-35-minute score includes 2070 seconds (00:30–35:00) for every run except run 19, which has 2040. Its missing valid window must not be silently counted as zero. Runs 12 and 35 had first observed task completions at 33.03m and 31.93m respectively. Thus the old pre-task-completion assumption does not hold for the entire expanded cohort.

The following comparisons use identical recorded reference functions; no reference refit is done for the shorter interval. All runs have the complete 00:30–30:00 interval scored (1770 seconds), preceding all first observed whole-task completions.

| Check | Best → worst run IDs |
|---|---|
| 35m, 85th reference | 3 → 5 → 12 → 6 → 4 → 36 → 35 → 18 → 1 → 8 → 19 → 7 → 2 → 9 |
| 35m, 90th reference | 3 → 5 → 12 → 6 → 36 → 4 → 35 → 18 → 1 → 8 → 19 → 7 → 2 → 9 |
| 35m, 95th reference | 3 → 5 → 12 → 6 → 36 → 4 → 18 → 1 → 35 → 8 → 19 → 7 → 2 → 9 |
| 00:30–30:00, 90th reference | 3 → 12 → 5 → 35 → 36 → 6 → 4 → 1 → 18 → 19 → 8 → 7 → 2 → 9 |

| Run | Mean units/s, 00:30–30:00 |
|---|---:|
| 03-s32-b8192-t1024 | 1.471 |
| 12-s32-b16384-t2048 | 1.426 |
| 05-s24-b16384-t1024 | 1.398 |
| 35-s32-b4096-t1024 | 1.337 |
| 36-s32-b16384-t1024 | 1.325 |
| 06-s24-b32768-t1024 | 1.324 |
| 04-s24-b4096-t1024 | 1.302 |
| 01-s24-b8192-t1024 | 1.269 |
| 18-s32-b8192-t2048 | 1.228 |
| 19-s32-b4096-t2048 | 1.221 |
| 08-s24-b8192-t2048 | 1.206 |
| 07-s24-b8192-t512 | 1.159 |
| 02-s8-b8192-t1024 | 0.841 |
| 09-s8-b4096-t2048 | 0.668 |

## Completion, quality and failures

Fastest time to eight completions: run 12, s32/b16384/t2048, 47m08.1s. Its first eight finishers passed only 1/8 supplementary checks. This is an end-to-end completion result and is not interchangeable with the computational work score.

Run 36, s32/b16384/t1024, reached eight at 68m52.3s and all 16 at 85m28.4s. Its first eight passed 1/8 supplementary checks; all 16 passed 5/16. All completed tasks passed the benchmark and original CLI checks, but supplementary failures remain. The original campaign QUALITY.md uses all 16 for run 36; the comparison table above consistently uses the first eight for every run.

Both s32/b32768 attempts failed with supervisor memory-pressure shutdown: run 13 at t2048 (MemAvailable 5.16 GiB, swap 16.0 GiB), run 34 at t1024 (MemAvailable 7.96 GiB, swap 15.835 GiB). They have no completed work-score measurement here. They are not demonstrated viable settings under the tested workload. Historical s24/b32768 did complete.

Run 9 has a severe profiler overrun. Run 36 has a smaller sampling-lag caveat (1.15s behind, 29 sampling errors). These remain documented in the mined manifest and campaign report.

## Interpretation and limits

Sequence count: s8 is consistently poor in this workload. s32 versus s24 is not uniformly positive: it helps at b8192, but the b16384/t1024 comparison favours s24. Budget: 16k is not a universal improvement; s32/t1024 peaks at 8k while s32/t2048 peaks at 16k among completed runs. Threshold: at s32, t1024 wins at 4k and 8k budgets, while t2048 wins at 16k. These reversals rule out multiplying independent parameter gains to predict the optimum.

These are empirical operation-rate units, not physical GPU utilisation or useful-task units. Context is an HTTP-overlap estimate, including queues and uniform output growth. Reprocessing earns computation credit. Fits are retrospective, use mixed operation windows and have no prospective repeated-run validation. Different trajectories, cache behaviour, concurrency and quality remain confounds. Do not call any configuration fully deterministic or fully quality-verified based on this campaign.

Recommendation under the user’s preferred work criterion: s32/b8192/t1024 is the measured leader to carry forward. s24/b16384/t1024 is a close historical alternative. Retain the distinction from the end-to-end completion leader and avoid a deployment change based solely on this report.

## Artifacts and reproduction

- [Smooth fits and detailed report](REPORT.md)
- [Scores](scores.csv)
- [Per-window calculations](scored-windows.csv)
- [Reference coefficients and validation](functions.json)
- [Mining manifest](../mined-final/manifest.json)
- [Handover and full methodology](../HANDOVER.md)

```bash
cd /home/jon/dual-spark-inference-kv-paging
python3 experiments/empirical-work-score/mine.py --campaign experiments/scheduler-completion-20260915 --window-seconds 30 --bucket-tokens 1000 --minutes 35 --reference-minutes 0 --output experiments/empirical-work-score/mined-final-reproduction
python3 experiments/empirical-work-score/fit_functions.py --samples experiments/empirical-work-score/mined-final-reproduction/samples.csv --quantile 0.90 --output experiments/empirical-work-score/smooth-final-reproduction
```

Output directories must be new. `final_summary.py` regenerates this additional synthesis using the fixed smooth-final and campaign paths. Arithmetic validation recomputed all exported scores, and checked complete common-window coverage. Source campaign audit reports all 14 completed measurement protocols passed; that does not certify quality or determinism.
