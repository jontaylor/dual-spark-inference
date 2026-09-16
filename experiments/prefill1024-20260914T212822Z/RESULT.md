Threshold1024 trial active

Both ranks now run --long-prefill-token-threshold1024 with --max-num-batched-tokens8192. Added optional launcher passthrough; only new configuration field is long_prefill_token_threshold=1024. Before/after launcher/config and actual runtime arguments saved alongside this report. API health200; no ERROR/Traceback in captured server logs.

16 concurrent synthetic raw-completion requests (1888–1889 prompt tokens, max16 generated tokens) completed.12 returned nonempty text;4 ended after one generated token without nonempty text. First-text times for12:22.98s (6),41.54s (2),46.61s (1),55.88–55.89s (3). The probe saved all results then its terminal summary sort failed on absent first-text timestamps; saved probe.json is intact. These timings are not a matched before/after comparison and include concurrent workload contention.

Server accounting shows256-token prefill chunks, consistent with existing fair-prefill budget8192 divided across32 requests. Threshold1024 is a ceiling; fair sharing can reduce it further. Staggered entry to generation persists. No aggregate-throughput or correctness-parity improvement claimed. Representative workload task notified to preserve settings and record comparison evidence. Trial left running as requested.
