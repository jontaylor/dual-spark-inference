# Final campaign closeout

Completed at 2026-09-15 22:35:20 UTC. User revisions applied: skip s8/s24/t512; no retries after arm35; final arm36 ran to all16 normal completions. Six requested successful configurations measured; both b32768 failures preserved.

| s32 configuration (budget / threshold) | First eight (s) | Generation tok/s through eight | Prompt reuse |
|---|---:|---:|---:|
| 03-s32-b8192-t1024 | 4414.6 | 256.53 | 95.58% |
| 12-s32-b16384-t2048 | 2828.1 | 247.31 | 96.02% |
| 18-s32-b8192-t2048 | 3326.4 | 252.21 | 97.06% |
| 19-s32-b4096-t2048 | 2996.4 | 268.62 | 96.15% |
| 35-s32-b4096-t1024 | 4015.8 | 202.29 | 95.64% |
| 36-s32-b16384-t1024 | 4132.3 | 247.14 | 96.30% |

Final arm36: all16 in 5128.4s from workload launch, 214.93 generated tok/s over its full measured window, 96.48% prompt reuse. Benchmark and original CLI:16 pass; supplementary:5 pass/11 fail. These measurements do not certify numerical determinism. Read QUALITY.md and quality-audit.json for individual verification evidence.

Both rank containers were independently inspected after completion; actual settings32/16384/1024 and Running=true saved in arm36/final-runtime-r*.json. API health200. Experiment controller and GPU watcher exited normally; report watcher completed analysis/audit.

The independent repeat-load controller automatically launched cycle027 after cycle026 finished. Its STOP mechanism was invoked to honor the final-test instruction, and controller exit/status stopped was verified. Cycle027 is an interrupted extra workload, excluded from all arm36 cutoff counters and comparisons; no model restart occurred. Evidence: final-load-controller-status.json and automatic-rollover-stop.json. Completed cycle026 manifest remains finished.

The six measurement audits pass, with the same frozen workload and only the requested scheduler settings changed. b32768/t1024 and b32768/t2048 attempts failed on the memory guard, not task quality. The final worker profile at120s reported1.15s sampling lag; attribution caveat remains in RESULTS.md.
