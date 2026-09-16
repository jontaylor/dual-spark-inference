Arm 2 failed before model loading: rank0 free space was below the unchanged 56 GiB cache-plus-reserve requirement. First-arm measurement preserved. Moved historical tensor trace files to rank1 with SHA256 verification; locations in experiments/targeted-determinism-20260913/relocations-20260915.json. Archive previous inactive campaign before startup so temporary archive files do not consume startup reserve. Archive step now recognizes and verifies already relocated archives on resume. Resume arm2 with identical s8/b8192/t1024 settings.


## Arm 13 memory-pressure failure

Rank 0 supervisor stopped serving at 1789492783.6102922 after five low-memory samples. Final available memory 5.16 GiB; swap occupied 16.0 GiB. Zero tasks completed. Both server and supervisor logs retained in arm 13; failed.json and controller-failure.json preserve outcome. No guard or server configuration change. Resume at 14 skips excluded arms 14–17 and tests arm 18 next. Failed arm 13 remains incomplete in audit; it is not a successful measurement.


## Arm 34 memory-pressure failure

Rank0 supervisor stopped serving at 1789501172.7062294 after five low-memory samples: available7.96GiB, free4.88GiB, swap used15.835GiB. Zero tasks completed. Both server/supervisor logs and controller failure preserved. Resume35 with unchanged guards and fixed settings. Arm34 remains incomplete.
