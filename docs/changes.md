# Configuration versus source changes

| Category | Examples | Maintained in |
|---|---|---|
| Standard configuration | Model/revision, BF16 KV, TP2, EP, MTP3, C4, context and batch budgets, memory fraction, graphs | `deploy_config.example.json`; ignored local configuration |
| Code with configuration switches | PLE hashing/cache/transport, tuned BF16 and MoE kernels, fair prefill, restricted drafting | `server/vllm/`, `server/gb10/`; selected via configuration |
| Correctness fixes | Sparse recurrent-state cleanup, 1,600-token group alignment, in-flight state budgeting | Focused server source commit and regressions |
| Deployment | Launchers, supervisor, unit rendering, model verification, preparation | This repository |
| Selected data | Draft token IDs, GEMM plans | `assets/` and `server/gb10/` |
| Large generated data | Model weights, packed table, binaries, compiler caches | Local storage, rebuilt from saved inputs |
| Evidence | Aggregate results and reproducible test/benchmark code | `docs/` and both repositories' tests |

The cache fix reclaims obsolete checkpoints inside the existing pool. It did
not increase the memory-utilisation setting to conceal the retained-block bug.
Full-context C4 validation after the fix observed 748/1035 peak sampled blocks,
724 during decode and zero preemptions.

The public repositories preserve the selected implementation and aggregate
findings. They do not publish the private workload corpus, attachments, raw
responses, API credential or the original experimental history containing those
outputs. Original local commit IDs remain recorded for provenance.
