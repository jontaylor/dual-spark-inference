# Quality of tasks completed at cutoff

Counts are pass / fail / unknown from archived operator verification. Completion remains the requested stopping criterion. These checks do not establish numerical determinism or a clean full test suite.

| Arm | Sequences / budget / threshold | Benchmark | Original CLI | Supplementary | Verified repair |
|---|---|---|---|---|---|
| 1 | 24 / 8192 / 1024 | 8 / 0 / 0 | 8 / 0 / 0 | 3 / 5 / 0 | 3 / 5 / 0 |
| 2 | 8 / 8192 / 1024 | 8 / 0 / 0 | 8 / 0 / 0 | 4 / 4 / 0 | 4 / 4 / 0 |
| 3 | 32 / 8192 / 1024 | 8 / 0 / 0 | 8 / 0 / 0 | 6 / 2 / 0 | 6 / 2 / 0 |
| 4 | 24 / 4096 / 1024 | 8 / 0 / 0 | 8 / 0 / 0 | 4 / 4 / 0 | 4 / 4 / 0 |
| 5 | 24 / 16384 / 1024 | 8 / 0 / 0 | 8 / 0 / 0 | 5 / 3 / 0 | 5 / 3 / 0 |
| 6 | 24 / 32768 / 1024 | 8 / 0 / 0 | 8 / 0 / 0 | 6 / 2 / 0 | 6 / 2 / 0 |
| 7 | 24 / 8192 / 512 | 8 / 0 / 0 | 8 / 0 / 0 | 6 / 2 / 0 | 6 / 2 / 0 |
| 8 | 24 / 8192 / 2048 | 8 / 0 / 0 | 8 / 0 / 0 | 1 / 7 / 0 | 1 / 7 / 0 |
| 9 | 8 / 4096 / 2048 | 8 / 0 / 0 | 8 / 0 / 0 | 7 / 1 / 0 | 7 / 1 / 0 |
| 12 | 32 / 16384 / 2048 | 8 / 0 / 0 | 8 / 0 / 0 | 1 / 7 / 0 | 1 / 7 / 0 |
| 18 | 32 / 8192 / 2048 | 8 / 0 / 0 | 8 / 0 / 0 | 6 / 2 / 0 | 6 / 2 / 0 |
| 19 | 32 / 4096 / 2048 | 8 / 0 / 0 | 8 / 0 / 0 | 2 / 6 / 0 | 2 / 6 / 0 |
| 35 | 32 / 4096 / 1024 | 8 / 0 / 0 | 8 / 0 / 0 | 7 / 1 / 0 | 7 / 1 / 0 |
| 36 | 32 / 16384 / 1024 | 16 / 0 / 0 | 16 / 0 / 0 | 5 / 11 / 0 | 5 / 11 / 0 |

Task identities, source paths and relative-path/CWD check details are preserved in [quality-audit.json](quality-audit.json). Different finishing subsets and a single run per configuration prevent attributing these quality differences to scheduler settings.
