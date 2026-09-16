# Cache route investigation

User snapshot: local lookup hit ratio 32.6%, external conditional ratio 93.9%, combined prompt reuse 95.9%. Scheduler external-query denominator is tokens remaining after local reuse. Approximately 32.6% of all prompt tokens use local reuse, 63.3% use the connector, and 4.1% require computation (lookup versus usage timing can prevent exact reconciliation).

External connector hits include GPU-resident completion snapshots and filesystem-backed pages. The live worker restore implementation was copied to rank_local_disk.live.py. In _transfer, a ResidentSlots entry is copied from GPU into CPU staging rather than read from a file; both paths then run fingerprint checks, copy staging to the destination GPU blocks, and optionally read the GPU bytes back and check again. Full readback is enabled in current configuration. This is materially different from direct local prefix block reuse, even without disk access. No direct GPU-only resident restore fast path is present in this code.

TransferResult reports filesystem bytes but elapsed time for the entire restore, including resident copies and checks. Therefore load_time_total must not be labeled pure disk-read latency. Filesystem reads use buffered I/O, so payload bytes do not establish physical NVMe read bytes.

Matched approximately 49-minute archived windows:

| Run | Local lookup hit % | External conditional hit % | Filesystem load GB | Cumulative restore job seconds |
| --- | ---: | ---: | ---: | ---: |
| H | 57.7 | 90.4 | 94.78 | 342.0 |
| I3 | 40.8 | 94.1 | 47.09 | 458.2 |
| J | 38.5 | 93.8 | 46.71 | 441.6 |

The earlier historical temperature sweep loaded 70.65 GB over its matched window, versus J's 46.71 GB. Its normalized archived metrics lack local/external prefix counter splits and restore timing. The user's remembered 82% local is not a verified matched-window value.

H contains a readback on/off experiment and different trajectories/concurrency. Cumulative job time can overlap other work and does not equal wall-clock serving time lost. These observations establish changed reuse routes and remaining restore overhead, not a causal estimate of throughput impact.

Next targeted optimization to evaluate: resident checkpoint reuse via safe block sharing where immutable, or GPU-to-GPU copies for private state, with equivalent integrity validation and deterministic resident/disk oracle checks. This requires code implementation and measured validation, not changing a hit-rate threshold. No live configuration or serving code was changed in this investigation.
