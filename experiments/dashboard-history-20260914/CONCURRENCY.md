# Historical concurrency follow-up

The current J run had substantially lower concurrency than the earlier temperature sweep. Across the previously selected approximately 49-minute windows, endpoint-weighted mean running requests were 9.43 earlier, 8.25 H, 9.42 I3 and 7.03 J. Estimated time with at least nine requests running was 62.9%, 40.3%, 56.3% and 24.0%, respectively. Thus concurrency plausibly contributes to the overall J throughput gap; an equal-mean argument alone was insufficient.

Matching observed request counts does not remove the gap. Using intervals whose running-request count agrees at both endpoints, keeping counts with at least 30 seconds of support in each run, and reweighting earlier per-count throughput onto the later run's observed interval durations:

| Later run | Earlier rate at later count weights | Later measured rate | Coverage of later window |
| --- | ---: | ---: | ---: |
| H | 178.2 tokens/s | 138.2 tokens/s | 44.1% |
| I3 | 192.2 tokens/s | 149.6 tokens/s | 46.4% |
| J | 174.5 tokens/s | 145.7 tokens/s | 38.9% |

Historical sampling was approximately five seconds versus ten seconds later. Repeating after retaining every second historical sample gives earlier reweighted rates of 171.9, 182.9 and 167.2 tokens/s respectively. The remaining J gap is approximately 13% under this sensitivity check, versus 16% with original sampling; I3 remains approximately 18–22% lower. These are descriptive subset comparisons, not causal estimates or whole-window counterfactual predictions.

Endpoint agreement does not establish constant concurrency inside an interval. Running requests include prefill, so equal counts do not establish equal decode batch sizes, context lengths, speculative acceptance, or memory traffic. Generated workload trajectories also differ. Subsetting stable endpoints can introduce selection bias. Historical counters do not include speculation or completed-cache usage measurements.

Conclusion: different concurrency levels explain some of the apparent decline, but observed request-count distributions alone do not explain all of it. Earlier attribution to deterministic kernels as the leading suspect was too strong without controlled comparison; these data do not isolate kernel costs. No serving settings, client workload or concurrency limits were changed for this analysis.

Reproduce with `python3 analyze_concurrency.py` and `python3 analyze_concurrency.py --ten-second`. JSON artifacts contain occupancy, interval support, per-count throughput, and weighting details.
