# GPU utilisation stall capture — 14 September 2026

Observed180s from17:14:56UTC (18:14:56London). Read-only workload observation; no restart, model/config/workload changes or inference probes. nvidia-smi sampled both ranks every200ms; nonblocking Python stack sampling25Hz on head worker/engine for180s and rank1 worker for100s. Sensor reporting repeats an internal interval, so repeated low samples do not establish precise GPU idle duration.

## Finding

Five single-digit-utilisation episodes were captured: one rank0 at1%, four rank1 at2%. Every episode had nearby eviction/disk-transfer bursts on both ranks. The worker main thread is synchronously blocked by eviction saves before reusing GPU pages. This custom native-pressure fence makes hashing and filesystem persistence part of the inference critical path.

Exact source chain: GPU runner prepare_completion → GB10AlignedOffloadingConnector.prepare_completion → OffloadingConnectorWorker.handle_preemptions → RankLocalDiskWorker.wait → Future.result → condition wait. Disk executor runs RankLocalDiskWorker._transfer: GPU→CPU staging, _fingerprints (SHA256), ContentAddressedPages.store / batch_store_block. It processes jobs through its single executor. Save acknowledgement comes after the full transfer/hash/store job. A dedup hit avoids writing payload but still stages/hashes it. Subsequent restores can add disk read, checksum, GPU copy and GPU readback verification.

The fence is needed to prevent overwriting a cached page before preserving it. Waiting for complete hashing/filesystem storage before allowing inference to proceed is the performance issue identified here; simply removing the fence would be incorrect.

## Captured episodes

Transfer totals below are per rank0 (rank1 has matching byte totals), over a correlation window from1s before first low sample until0.5s after last low interval. They are filesystem payload, decimalMB, not NVMe-device I/O. This window includes nearby work and is not a claim every byte was transferred inside an exact hardware stall.

| London time | GPU rank low | Utilisation | Stored MB/rank | Loaded MB/rank |
|---|---|---|---|---|
| 18:15:01 | 1 | 2% | 570.29 | 27.16 |
| 18:15:11 | 1 | 2% | 325.88 | 244.41 |
| 18:16:10 | 0 | 1% | 380.19 | 135.78 |
| 18:16:16 | 1 | 2% | 380.19 | 0.00 |
| 18:17:33 | 1 | 2% | 380.19 | 244.41 |

At18:16:10,15 stores and one restore are reported per rank nearby:380.19MBnew stores plus135.78MBreads. Summed rank0 transfer-job duration0.620s; rank1 0.598s. At18:16:17 there are380.19MBstores and no disk reads, so reads alone cannot explain these episodes.

## Stack evidence

Head worker:203 samples ×40ms =8.12 weighted seconds in prepare_completion→handle_preemptions→wait→Future.result, out of174.48 seconds of successfully sampled main-thread stacks:4.65%. Additional event capture0.68s. The kv-disk thread sampled5.36s fingerprinting,2.68s batch_store_block,4.12s CUDA synchronization. These categories overlap main-thread waiting; do not add them as separate walltime costs.

Rank1:3.44/97.32 sampled main-thread seconds (3.53%) in the same eviction fence. Its disk thread sampled2.80s fingerprinting,1.04s batch_store_block,2.20s synchronization.

Head worker ordinary output synchronize dominates overall stack samples (132.92s), but that usually means waiting for real GPU work, not evidence all that time is a GPU stall. Tensor-parallel ranks must progress together; asymmetric transfer/host readiness can make one rank fall idle before the other. The precise rank-to-rank collective timeline was not captured.

## Metrics and limitations

Valid metrics span150.55s: 175.50 aggregate generated tokens/s;9–12 running requests; zero preemptions;8.147GBstored and3.639GBloaded across both ranks.

Initial unprivileged /proc/io reads failed; those rows are preserved in metrics.jsonl and excluded. A separate privileged read-only collector supplies metrics-valid.jsonl. nvidia-smi and stack capture continued throughout. Nonblocking py-spy reported137 sampling errors on head worker and480 on engine; missing samples shorten speedscope weighted time. Consequently stack samples are used for aggregate attribution, NOT reconstructed exact wallclock correlation. Correlation timestamps come from GPU CSV and timestamped server logs. No native CUDA/NCCL timeline or controlled profiler-overhead benchmark was collected. Five captured episodes do not prove every possible future low reading has this cause.

## Fix target

Separate preservation of soon-to-be-reused GPU bytes from CPU hashing/persistence. Batch those copies into bounded staging buffers; release the model fence once source bytes are safely staged, then hash/write asynchronously with explicit backing states and backpressure if staging capacity is exhausted. Retain checksums and two-rank ownership safety. This needs implementation and correctness/performance validation; no change was made during this diagnostic. GPU restore traffic can still stall and must be measured separately after fixing store-side serialization.

Raw artifacts: gpu-r0/r1.csv, r0/r1.log, worker/engine/worker-r1.speedscope.json, profile-summary.txt, correlation.json, metrics-summary.json and collection scripts. Profilers and metric collectors completed. The copied remote py-spy executable is a temporary diagnostic file, not a serving override.
