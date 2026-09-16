# MTP depth selection audit

The retained MTP5 configuration has correctness validation but no isolated performance win over MTP3/MTP4 established by the overnight evidence. Initial candidate I changed depth 3 to 5 and graph sizes; the eventual serving candidate I3 additionally required page geometry/scheduler changes. H versus I3 completed-cycle throughput was 125.48 versus 126.34 tokens/s (about 0.7% difference); partial matched-window comparisons had differing actual concurrency, trajectories, cache state, and H readback trials. No controlled MTP4 comparison was located in this campaign. Keeping MTP5 as the selected configuration was not justified as a demonstrated depth optimum.

User's historical per-position accepted fractions: 0.79, 0.63, 0.50, 0.40, 0.31. Interpreting these as accepted-at-position / all draft attempts (vLLM cumulative survival fractions), expected emitted tokens per speculative iteration, including one target correction/bonus token and ignoring EOS/output truncation, are:

- MTP3: 1 + .79 + .63 + .50 = 2.92
- MTP4: 3.32, or 13.70% more than MTP3
- MTP5: 3.63, or 9.34% more than MTP4 and 24.32% more than MTP3

For equal batch/workload and unchanged acceptance curve, MTP5 beats MTP3 only if mean iteration cost grows less than 24.32%; MTP5 beats MTP4 only if it grows less than 9.34%. These thresholds are not predictions of actual cost. Going from depth 3 to 5 performs two additional draft steps and expands nominal target verification rows per request from 4 to 6; neither implies proportional elapsed-time cost.

The initial 130-second MTP5 workload acceptance curve was 75.14%, 55.74%, 41.38%, 32.91%, 26.14% (I3-initial-acceptance.json). It showed no improved draft accuracy relative to the user's historical figures, but workload differences prevent a controlled accuracy comparison.

The user's hypothesis is mechanistically plausible: extra fixed cost per iteration can make deeper speculation look preferable by amortizing that cost over more emitted tokens. It is not true for every kind of slowdown: work proportional to verified rows or draft depth can instead penalize MTP5. The checkpoint shadow cost is triggered per crossed 64-token boundary and must not be treated as a fixed per-iteration cost automatically amortized by MTP5.

Correct decision sequence: address/profile the newly introduced blocking decode/checkpoint paths, then compare MTP3/4/5 with matched request replay, contexts, concurrency and cache state, recording iteration time and acceptance as well as aggregate throughput. Historical MTP3/MTP4 evidence remains the prior; MTP5 remains an unproven candidate, not a demonstrated upgrade. No serving changes were made for this audit.
