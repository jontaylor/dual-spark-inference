# Historical dashboard comparison

The dashboard points to a real lower throughput result relative to the earlier baseline. The previous H/I3/J comparison omitted that faster baseline and was insufficient for assessing the overall regression.

## Matched first approximately 49 minutes

| Run | Aggregate output tokens/s | Mean running requests | Written payload GB |
|---|---:|---:|---:|
| Yesterday, before deterministic rollout | 178.08 | 9.42 | 370.32 |
| H | 133.15 | 8.24 | 80.52 |
| I3 | 143.52 | 9.40 | 127.42 |
| J | 131.83 | 7.01 | 82.88 |

The yesterday comparison is the September 13 18:21 UTC temperature sweep, corresponding to 19:21 London if the dashboard uses local time. Earlier peaks align with other campaign launches (11:07 UTC tool-role, 13:14 independent-worker, 16:49 environment-info); those are different experiments and are not used for the table.

The archived server counter covered 580839 generated tokens across 3883.25 seconds, yielding 149.58 tokens/s over yesterday's whole cycle. H and I3 completed at 125.48 and 126.34 respectively. J was incomplete at the observation cutoff and should not receive a whole-cycle comparison. Completed-response evidence gives yesterday 95.33% prompt reuse across 580 requests; archived server samples did not record the cached-usage counter. Missing archived metrics, including preemptions, request-success and speculative counters, are null rather than zero.

## Workload controls actually checked

The older sweep and J have identical current-source hash manifests (128 entries), historical-source manifests (103 entries), source revisions, condition definitions excluding their directory paths, sampling conditions, output limits, eight arms and four permitted inference slots per arm. See `workload-provenance.json`.

This establishes the same frozen workload definition, not identical generated trajectories, context histories or instantaneous overlap. Nevertheless, yesterday and I3 had almost identical mean running-request counts (9.42/9.40), while I3 throughput was 19.4% lower. Reduced average concurrency alone does not explain that comparison. Matching means does not control concurrency distribution, context length or prefill/decode mixtures.

## Serving changes and attribution limits

The faster period predates the recorded deterministic-kernel rollout. The preserved pre-isolation configuration at 20:05 UTC already used MTP3, TP2, BF16 KV, 32 sequences, 8192 batched tokens, 40 GiB GPU KV and 48 GiB logical disk per rank, with transfer verification enabled. It had only the FLA threshold runtime override, rather than the subsequent fixed-reduction GEMMs and GDN/QSA/RMS execution changes. This is a nearby configuration snapshot, not an independent live hash audit of the earlier campaign start; absent fields must not be treated as proven runtime values.

Later changes also include aligned completion captures, content deduplication, inherited page sharing, MTP5/page-layout changes and smaller reservations. The deterministic computation routes are a leading candidate for the remaining throughput cost, but their isolated contribution has not been measured. Earlier writes were much higher (370.32 GB in the comparable window), so disk volume by itself is not an adequate explanation of the lower later throughput.

A causal test must compare the original fast computation path and current deterministic path on an identical replay, holding cache policy, speculative depth, prompt lengths and overlap fixed. This inspection made no serving or workload changes and did not restart the model. No claim is made that all of the 19–26% observed gap can be recovered while preserving determinism.

Evidence: `yesterday-temperature-counters.json` (filtered archival server samples with explicit presence), `comparison.json`, `workload-provenance.json`, `configuration-snapshot-comparison.json`, and the prior final H/I3/J metric windows.
