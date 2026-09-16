# Completed scheduler campaign

All eight configurations ran in the agreed order, each with a fresh frozen campaign and a 600-second measured load window. The configuration/collection audit passed for all eight: 300 metric samples per arm, maximum gap about 2.001seconds, both ranks’ hardware data, 32 saved Python profiles overall, readable campaign archives and matching frozen workload specifications.

The highest observed aggregate generation throughput was 32 sequences / 4096 total tokens / 2048 prefill threshold: 327.16 tokens/s for the full window, 355.59 tokens/s in the final five minutes, 3.29 seconds mean recorded TTFT.

Across the four matched configuration pairs, 32 versus 24 sequences increased observed generation throughput by 5.1–14.6%; threshold 2048 versus 512 increased it by 8.9–24.1%. Raising the total budget 4,096 → 16,384 was mixed:−9.9% to+2.7%. These observations favour 32 sequences and 2048 threshold for the next validation; they do not establish a universal optimum or a significant advantage of 4,096 over 16,384.

Important limits: one run per configuration, no drift-repeat, different generated trajectories/prompt work/cache reuse, and only ten minutes per run. All 16 benchmark arms were still marked running at export in every configuration. Completed inference requests are not completed benchmark tasks. No model-quality or numerical-determinism claim. Python sampling had missed stack samples, recorded in logs; no intrusive GPU kernel tracing was performed.

No server ERROR/Traceback appeared in the captured measured-run logs, and reported preemptions were zero for all eight arms. There was one arm 6 startup preflight failure due to insufficient disk reserve before measurement; verified archive relocation resolved it without changing serving settings. See RECOVERY.md and archive-relocations.json.

Current service: final matrix configuration 24 sequences / 16,384 budget / 2048 threshold, running on both ranks with HTTP 200 health. All mounted runtime-override hashes were verified on both ranks. MTP3, PLE 1 GiB budget, 16 gather threads and both NCCL HCAs remain in place. The campaign controller and GPU watcher have exited; the user-owned repeat workload controller remains untouched.

Evidence: COMPARISON.md for all eight results and 12 matched pairs; RESULTS.md for throughput/cache/latency/cutoff tables; results.json for counters, profiles, GPU data and workload states; audit.json for collection checks; final-live.json for actual final runtime. Each numbered directory contains the raw measured-window artifacts.
