# Scheduler completion campaign

Authorized scope: all 36 combinations of max-num-seqs {8,24,32}, max-num-batched-tokens {4096,8192,16384,32768}, long-prefill-token-threshold {512,1024,2048}. User confirmed b32 means 32768.

Every arm restarts both ranks using the existing restart-reset workload controller and the same frozen 16 tasks. Sampling, retry policy, inference slots, MTP3, numerical overrides, cache policy, PLE, NCCL and all non-scheduler configuration remain at the saved baseline. Both actual container argument lists are captured and asserted for each arm.

Primary endpoint: elapsed time from fresh campaign launch until first observed eighth task with run.json status complete. Also report from first observed active inference. Poll completion about every six seconds (two-second telemetry loop); retain observations to bracket the crossing. Complete means harness returned normally, not quality-test success. Incomplete and interrupted do not count. Record task identities, failures, quality evidence and work/reuse counters. There is no ten-minute measurement cutoff. More than eight failed/interrupted tasks makes the endpoint unattainable and stops the campaign for investigation.

First eight runs estimate each factor against the common reference (24,8192,1024). Later runs maximize new pairwise coverage, then the number of comparisons differing in only one setting. All combinations remain in scope. Single observations and changing agent trajectories limit statistical conclusions; eighth-task identities and quality must accompany timing. PLE filesystem cache is not flushed.

Collect two-second metrics, five-second hardware, half-second GPU samples, and nonblocking Python profiles at 120/420 seconds. Preserve pre-cutoff task snapshot and exact cutoff counters before evidence export; subsequent restart cancels the other tasks through existing machinery. Archive previous inactive campaigns with all-file hash verification, move archives to rank1 with SHA256 verification, then remove source copies. Never remove the frozen parent or a live campaign. Final configuration remains running.

Startup alone is approximately six hours for 36 arms; workload duration is additional and will be estimated from observed completion times. Stop on infrastructure failure, inspect authoritative process state and resume without repeating completed arms. No automatic quality or determinism claims.

| Order | max-num-seqs | batch tokens | prefill threshold |
|---:|---:|---:|---:|
| 1 | 24 | 8192 | 1024 |
| 2 | 8 | 8192 | 1024 |
| 3 | 32 | 8192 | 1024 |
| 4 | 24 | 4096 | 1024 |
| 5 | 24 | 16384 | 1024 |
| 6 | 24 | 32768 | 1024 |
| 7 | 24 | 8192 | 512 |
| 8 | 24 | 8192 | 2048 |
| 9 | 8 | 4096 | 2048 |
| 10 | 8 | 16384 | 512 |
| 11 | 32 | 4096 | 512 |
| 12 | 32 | 16384 | 2048 |
| 13 | 32 | 32768 | 2048 |
| 14 | 8 | 32768 | 512 |
| 15 | 8 | 4096 | 512 |
| 16 | 8 | 8192 | 512 |
| 17 | 8 | 8192 | 2048 |
| 18 | 32 | 8192 | 2048 |
| 19 | 32 | 4096 | 2048 |
| 20 | 32 | 8192 | 512 |
| 21 | 32 | 16384 | 512 |
| 22 | 32 | 32768 | 512 |
| 23 | 24 | 32768 | 512 |
| 24 | 24 | 4096 | 512 |
| 25 | 24 | 16384 | 512 |
| 26 | 24 | 4096 | 2048 |
| 27 | 24 | 16384 | 2048 |
| 28 | 24 | 32768 | 2048 |
| 29 | 8 | 32768 | 2048 |
| 30 | 8 | 16384 | 2048 |
| 31 | 8 | 16384 | 1024 |
| 32 | 8 | 4096 | 1024 |
| 33 | 8 | 32768 | 1024 |
| 34 | 32 | 32768 | 1024 |
| 35 | 32 | 4096 | 1024 |
| 36 | 32 | 16384 | 1024 |

## User scope revision

Skip all s8, s24 and t512 configurations. Preserve original numbering/order and completed historical evidence. Arm 10 is abandoned as a partial measurement by user instruction. The requested scope is now eight s32 configurations (four budgets × thresholds 1024/2048); arm 3 is already complete, leaving seven runs. See scope-revision.json and individual skipped.json files.

## Final-run revision

User requested arm35 finish at8, then arm36 (s32/b16384/t1024) run last to all16 normal task completions, without a time cutoff. No retries of failed13/34. Preserve those failures in reports. Final server remains running. Compare all arms through eighth completion, and report arm36 full-run results separately. final-run-policy.json defines authoritative revised targets and required IDs.
