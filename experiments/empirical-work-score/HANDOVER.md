# Handover: empirical context-adjusted inference work scoring

Written 2026-09-15. Workspace: `/home/jon/dual-spark-inference-kv-paging`.

This work was done in a side conversation. It is offline analysis only. It did not change serving configuration, restart a server, control workloads, or modify the main campaign controller. Do not interpret this document as authorization to change the campaign or its stopping rule.

## Purpose and decision reached

The main experiment measures scheduler configurations using a fresh 16-task agent workload and stops each run at the first observation of eight normally completed tasks. The scheduler factors are max sequences, maximum batched tokens, and long-prefill threshold. `b32` means 32,768 tokens.

Time to eight completions is a useful end-to-end outcome but does not isolate inference execution speed. Even with temperature zero and work on within-lifecycle determinism, different restarted configurations produced different model/task trajectories, prompt lengths, output lengths, and verification results. Cross-configuration trajectory equality is not established.

The user wants an estimate of computational work processed, combining decode and uncached prefill, with context-dependent valuation. They accept an empirical relative reference rather than a physical hardware cost model. Their preferred comparison is the cumulative first 35 minutes, before any whole task finished in the original eight-run cohort.

Current approach: fit smooth upper-quantile throughput functions directly to archived observations, then divide each measured operation rate by its fitted reference at the estimated context. Add decode and prefill units, and average over the selected time interval.

This credits computation, including repeated work. It does not measure task usefulness, correctness, physical GPU utilisation, or proven trajectory-independent hardware efficiency.

## Files and current authoritative artifacts

All paths in this table are relative to this directory.

| File/directory | Purpose |
|---|---|
| `mine.py` | Reads completed campaign archives; reconstructs context; calculates measured rates; builds original bucket-peak reference, plots and scores. Produces the raw samples used by the smooth fitter. |
| `fit_functions.py` | Fits smooth references from sample CSV; performs validation; exports callable scoring class, fitted model and new report. |
| `all-reference-v2/samples.csv` | Frozen input cohort for the current smooth report: runs 1–9, 1,267 30-second windows. |
| `all-reference-v2/manifest.json` | Source archive paths, request exclusions, campaign windows, profiler caveats and mining arguments. |
| `all-reference-v2/REPORT.md` | Historical bucket-peak report; superseded as a scoring reference by the smooth report. |
| `smooth-v2/REPORT.md` | Current smooth-fit report: methods, ranked results, validation and sensitivity. |
| `smooth-v2/curves.png`, `curves.svg` | Raw scatter, smooth reference, median diagnostic, and 85–95% quantile sensitivity band. |
| `smooth-v2/functions.json` | Exact coefficients, domains, selected basis, validation folds, diagnostic/sensitivity fits, source CSV hash and fitter script hash. |
| `smooth-v2/reference-curves.csv` | Sampled numerical reference functions. |
| `smooth-v2/scored-windows.csv` | Per-window rates, reference values, operation units and combined score. |
| `smooth-v2/scores.csv` | Interval summaries with decode/prefill components and alternative reference quantiles. |
| `first35m-v1/` | Initial exploratory peak reference restricted to first 35 minutes; not the current reference. |
| `smooth-v1/` | Incomplete failed first fitting attempt: tiny negative floating-point prefill rates caused scoring to reject an input. Do not use as a finished report. Fixed in v2. |

At handover, this analysis directory is untracked in git; no commit has been made. The script defaults point at the frozen v2 artifacts, not automatically at newer reports.

## Why this analysis was introduced

Across the original eight completed runs, actual completion order was:

`8 → 5 → 6 → 1 → 2 → 3 → 4 → 7`

Ranking by cumulative generated tokens gave:

| Checkpoint after workload launch | Predicted order | Correct pairs / 28 | Spearman correlation |
|---|---|---:|---:|
| 15m | 8 → 5 → 3 → 6 → 1 → 7 → 4 → 2 | 22 | 0.714 |
| 20m | 8 → 5 → 3 → 6 → 1 → 4 → 7 → 2 | 23 | 0.738 |
| 25m | 8 → 3 → 5 → 1 → 6 → 4 → 7 → 2 | 21 | 0.619 |
| 30m | 3 → 8 → 5 → 1 → 6 → 4 → 7 → 2 | 20 | 0.500 |

Run 3 generated more tokens by 30 minutes than run 8, and continued generating faster in minutes 30–50, but had zero completed tasks at minute 50 versus run 8's seven. Its total generation at cutoff was 1,129,951 versus 809,334 for run 8. This demonstrates that token volume and task completion are distinct measures; it does not establish why the trajectories differed.

Earliest whole-task completion across those eight runs: run 8, `baseline-restricted-r04`, task duration 2,145.377s (35m45.4s); first observed 2,152.455s after workload launch. Thus the 35-minute comparison precedes whole-task dropout in that cohort. It does not guarantee all 16 tasks continuously offered inference work. Recheck earliest completion for newly added runs.

Quality is separate: run 3's cutoff finishers passed 6/8 supplementary checks; run 8's passed 1/8. Both passed 8/8 benchmark and original CLI checks. These are different finishing subsets. No scheduler-quality causation or fully verified deterministic serving claim follows.

## Inputs and mining calculation

Campaign root: `../scheduler-completion-20260915/`.

For each completed arm, `mine.py` reads:

- `complete.json` as the completion eligibility marker.
- `window.json`: workload launch is `controller_cycle_start`, not the later first-observed inference timestamp.
- `metrics.jsonl.gz`: timestamped Prometheus samples, usually every two seconds.
- `campaign-evidence.tar.gz`, specifically `speculative-requests.jsonl`: per-request timestamps and provider usage.
- `profile-*-intervention.json`: caveats copied into the manifest.

Only arms with both `complete.json` and the evidence archive are included. It does not fetch data from hosts or contact inference endpoints. New/current runs without a completed archive are listed as excluded.

For every non-overlapping 30-second window aligned to workload launch:

1. Interpolate cumulative server counters linearly at window endpoints.
2. Decode rate = delta `vllm:generation_tokens_total` / window seconds.
3. Uncached prefill rate = (delta `vllm:prompt_tokens_total` − delta `vllm:prompt_tokens_cached_total`) / window seconds.
4. Reconstruct context from overlapping HTTP requests as below.

Counter series sum all labels with the exact metric name. These archives expose one logical engine; audit this assumption if using a future multi-engine dataset. The miner rejects counter resets and non-increasing timestamps. It uses complete windows only.

### Context reconstruction

For a request with start S, end E, prompt tokens P and output tokens O:

`estimated_context(t) = P + O * (t-S)/(E-S)` for S <= t < E.

At each instant, take the arithmetic mean over overlapping HTTP requests. Integrate that mean over the window and divide by the duration for which at least one reconstructed request is present. The integration splits at request boundaries; the midpoint calculation is exact for this piecewise linear estimate.

Requests are deduplicated by client request ID. Missing/invalid timing or usage is excluded and counted. This assumes IDs are present and unique in the source.

This is NOT an observed GPU-context gauge. It includes queue/prefill time, assumes uniform output growth, weights HTTP requests equally, and may omit calls still unfinished at archive capture. Both decode and prefill use this same estimate. Context coverage measures temporal coverage, not completeness of the request population.

Windows need >=95% temporal context coverage. They also need nonnegative uncached deltas within the miner's tiny numerical tolerance. Raw samples preserve validity and coverage. The smooth fitter clamps a negative prefill rate with magnitude <1e-6 tok/s to zero; it does not silently clamp substantial negatives.

### Original peak calculation (superseded)

Assign context to `floor(context/1000)*1000`. Take maximum observed decode rate and maximum observed prefill rate in each bucket independently. Score by the sum of rate/peak ratios. Maxima can come from different windows. This yielded jagged curves and sparse-bucket sensitivity, motivating the smooth fitter. Peak source windows remain auditable in `peaks.csv`.

## Smooth reference fitting

Current cohort is frozen runs 1–9; references use all available archived windows, not just the first 35 minutes. The miner emitted 1,258 valid-context windows out of 1,267. Smooth decode fitting uses 1,258 positive-rate samples; prefill uses 1,226 positive-rate samples.

For each operation separately:

- Fit the 90th conditional percentile of log throughput versus transformed context.
- Each run contributes equal total fitting weight, so a longer run does not dominate by having more samples.
- Use positive rates for fitting active-operation capacity. Zero rates still earn zero units during scoring.
- Compare five candidate families: linear and quadratic polynomials in log context; cubic splines with 0, 2 or 4 interior knots.
- Leave one run out at a time. Calculate quantile (pinball) loss in log-throughput space on the held-out run. Average losses equally across runs.
- Select the simplest candidate within one standard error of the lowest validation loss. This is a simplicity heuristic, not proof that candidates are statistically equivalent.
- Refit the selected family using all runs. Also fit the median and 85th/95th percentiles using the selected family as diagnostics/sensitivity.

Quantile loss for residual e and quantile q: `max(q*e, (q-1)*e)`.

Basis definition and exact coefficients are in `functions.json`. Domain normalisation uses the full observed context range; held-out throughput values do not enter training fits. These models have not been prospectively validated on new configurations. No finish-time rankings were used to select the curves.

Both selected models are linear in log-transformed context. More complex candidates had slightly lower losses, but insufficient improvement under the simplicity rule. Held-out observations below the selected reference were 89.5% for decode and 89.3% for prefill.

### Current functions

Approximate readable forms (use stored coefficients for exact calculations):

```python
decode_per_second(c)  = 551.76831581524 * (1 + c/1000)**(-0.15318626775355318)
prefill_per_second(c) = 10155.287184902083 * (1 + c/1000)**(-0.8212882468537027)

decode_units(c, d)  = d / decode_per_second(c)
prefill_units(c, p) = p / prefill_per_second(c)
combined_units_s   = decode_units(c, d) + prefill_units(c, p)
```

The actual implementation clamps context separately for each function:

- Decode: 2,935.269 to 64,440.474 tokens.
- Prefill: 3,080.399 to 64,440.474 tokens.

Outside these ranges, return the nearest boundary prediction; do not interpret that as validated extrapolation. Invalid/negative context or rates raise ValueError. The methods take scalar inputs.

Units above 1 are allowed. These are relative computational indices, not physical cluster utilisation; two separately fitted conditional reference rates do not imply a jointly achievable hardware ceiling.

For an interval, integrate each window score times its duration. Mean score = integrated units / scored seconds. Do NOT evaluate the formula once using the interval's average context and average rates: the nonlinear normalisation must be applied per window first.

## Results most relevant to the user

User prefers cumulative 0–35-minute work over the isolated 30–35-minute interval. In the current smooth report, all runs have exactly 2,070 scored seconds: 00:30–35:00. The initial 30s is excluded for insufficient reconstructed context coverage. The smooth fitter recovers one additional near-zero-prefill window that the old peak scorer could not score because its bucket reference denominator was zero.

| Run | Sequences / budget / threshold | Mean combined units/s, first 35m |
|---|---|---:|
| 1 | 24 / 8192 / 1024 | 1.311834 |
| 2 | 8 / 8192 / 1024 | 0.858609 |
| 3 | 32 / 8192 / 1024 | 1.507439 |
| 4 | 24 / 4096 / 1024 | 1.343013 |
| 5 | 24 / 16384 / 1024 | 1.446584 |
| 6 | 24 / 32768 / 1024 | 1.395629 |
| 7 | 24 / 8192 / 512 | 1.191757 |
| 8 | 24 / 8192 / 2048 | 1.304384 |
| 9 | 8 / 4096 / 2048 | 0.726427 |

Ordering: **3 → 5 → 6 → 4 → 1 → 8 → 7 → 2 → 9**.

Interpretation: s8 is consistently poor for aggregate work in these observations. The direct s24/s32 comparison at b8192/t1024 favours s32 by 14.9%. At s24/t1024, batch budgets show no monotonic relationship: b16384 leads, and all four budgets fall within about 10% of the best. t1024 and t2048 are nearly tied at s24/b8192; t512 trails. These are single-run comparisons with an approximate score, not established causal effects.

The 30–35m ordering differs: **8 → 6 → 5 → 3 → 4 → 1 → 7 → 9 → 2**. At the 85th reference quantile it becomes **6 → 8 → 5 → 4 → 3 → 1 → 7 → 9 → 2**; at the 95th it matches the 90th order. Use the score CSV's sensitivity columns for interval-specific comparisons.

Run 9 has a documented profiler overrun: a requested 20s worker capture was still running after 156s at high CPU usage. It remains included, with the caveat inherited from the source manifest. Do not present its result as an unconfounded scheduler comparison.

## Reproduce the current smooth report exactly

Dependencies already available when run: Python 3, NumPy, SciPy 1.11.4, Matplotlib 3.6.3. No GPU dependencies. Check availability without changing the environment:

```bash
cd /home/jon/dual-spark-inference-kv-paging
python3 -c 'import numpy, scipy, matplotlib; print(numpy.__version__, scipy.__version__, matplotlib.__version__)'
```

Use the frozen raw input to keep the reference cohort unchanged. Output directories must NOT already exist; scripts refuse overwrite.

```bash
python3 experiments/empirical-work-score/fit_functions.py \
  --samples experiments/empirical-work-score/all-reference-v2/samples.csv \
  --quantile 0.90 \
  --output experiments/empirical-work-score/smooth-reproduction-01
```

If that destination exists, choose another new name. The command creates report, PNG/SVG plots, functions JSON, reference grid, scored windows and summary CSVs. It does not alter the original report.

## Refresh from newly completed campaign runs

Refreshing the cohort changes the reference. All compared runs must be rescored together; do not mix scores from different fitted reference versions.

```bash
cd /home/jon/dual-spark-inference-kv-paging
python3 experiments/empirical-work-score/mine.py \
  --campaign experiments/scheduler-completion-20260915 \
  --window-seconds 30 \
  --bucket-tokens 1000 \
  --minutes 35 \
  --reference-minutes 0 \
  --output experiments/empirical-work-score/mined-refresh-01

python3 experiments/empirical-work-score/fit_functions.py \
  --samples experiments/empirical-work-score/mined-refresh-01/samples.csv \
  --quantile 0.90 \
  --output experiments/empirical-work-score/smooth-refresh-01
```

`--reference-minutes 0` means all recorded windows. Set it to 35 to fit references using only the first 35 minutes, if explicitly desired; that is a different reference experiment. `--bucket-tokens` affects only the original peak reports: the smooth fitter ignores bucket peaks and fits the raw sample columns.

Inspect the new manifest's included and excluded runs, invalid request counts and profiler caveats. Review sample coverage and scored seconds for each run. Newly completed arms are discovered automatically; incomplete arms are skipped. No controller restart or network access is needed.

Important implementation limits: `fit_functions.py` currently hardcodes score intervals 0–35m and 30–35m. Its source cohort path defaults to frozen `all-reference-v2/samples.csv`. Always pass explicit paths for refreshed data. The miner always includes a 30–35m score row even if a shorter reference horizon leaves it uncovered. The fitter requires enough positive observations and multiple runs for meaningful cross-validation; it was exercised on this nine-run dataset, not arbitrary sparse datasets.

## Call the functions directly

```bash
cd /home/jon/dual-spark-inference-kv-paging
python3 - <<'PY'
import runpy
module = runpy.run_path('experiments/empirical-work-score/fit_functions.py')
f = module['WorkFunctions'](
    'experiments/empirical-work-score/smooth-v2/functions.json'
)
context = 32000
measured_decode = 250.0
measured_prefill = 200.0
print('Decode reference tok/s:', f.decode_per_second(context))
print('Prefill reference tok/s:', f.prefill_per_second(context))
d = f.decode_units(context, measured_decode)
p = f.prefill_units(context, measured_prefill)
print('Decode units/s:', d)
print('Prefill units/s:', p)
print('Combined units/s:', d+p)
PY
```

Loading with `runpy.run_path` does not run the fitting CLI because the main guard is not activated. Supplying the JSON path explicitly avoids accidentally using the default frozen reference for new analyses.

## Checks performed and quick verification

Completed checks:

- Synthetic single-request and overlapping-request context integration.
- Missing-context handling and linear counter interpolation.
- Positive finite fitted rates over a context grid.
- `units(context, reference(context)) == 1` and zero rate gives zero units.
- Boundary clamping and rejection of negative input rates.
- Independent recomputation of every exported smooth window score.
- Visual inspection of both generated curves.

Repeat the main score identity check against any output directory:

```bash
cd /home/jon/dual-spark-inference-kv-paging
python3 - <<'PY'
import csv, pathlib, runpy
out = pathlib.Path('experiments/empirical-work-score/smooth-v2')
m = runpy.run_path('experiments/empirical-work-score/fit_functions.py')
f = m['WorkFunctions'](out/'functions.json')
rows = list(csv.DictReader((out/'scored-windows.csv').open()))
for r in rows:
    c = float(r['context'])
    expected = (f.decode_units(c, float(r['decode_tok_s']))
                + f.prefill_units(c, float(r['prefill_tok_s'])))
    assert abs(expected - float(r['combined_units_s'])) < 1e-10
print('Verified windows:', len(rows))
PY
```

These checks verify implementation arithmetic, not the accuracy of the context approximation or causal validity of the score. There is no permanent automated test suite; the checks above were executed as standalone snippets.

## Remaining uncertainties and sensible next steps

1. Compare future configurations using the same saved reference or refit/rescore the entire cohort together. State which policy was used.
2. Recheck first task completion and source coverage when adding runs; 35m is not guaranteed pre-completion forever.
3. Retain decode/prefill components, scored seconds and reference sensitivity rather than only one headline number.
4. A fitted reference is not a physical saturation curve. It absorbs queueing, concurrency, operation mix and trajectory differences present in the observations.
5. Uniform output growth and HTTP-overlap context are the largest measurement approximations. Direct scheduler-level context/token telemetry would improve a future campaign, but no serving instrumentation changes were made or authorized here.
6. An early-exit policy was discussed but NOT implemented. No instruction to stop the main campaign follows from these scores.
7. If changes to scoring scripts are made, write a new output directory and preserve existing artifacts. Do not overwrite source archives or campaign reports.

## Final analysis update

The final frozen cohort now contains 14 completed runs. See [final synthesis](smooth-final/FINAL_ANALYSIS.md), [fit report](smooth-final/REPORT.md), and [validation](smooth-final/final-validation.json). Inputs are in mined-final; models and scored data are in smooth-final. The historical v2/refresh reports above remain preserved. Regenerate the final synthesis with `python3 experiments/empirical-work-score/final_summary.py` from the workspace root. The final measured leader remains s32/b8192/t1024; the predicted s32/b16384/t1024 advantage did not materialise. Runs 12 and 35 complete tasks before 35m; final analysis therefore includes a complete common 00:30–30:00 comparison.
