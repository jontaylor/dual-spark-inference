# Scheduler campaign

Eight arms in matrix.json, each restarted on both ranks. Keep MTP3, fair-prefill, numerical kernels, cache policies, graph-size list, PLE and NCCL settings fixed. Only max_num_seqs, max_num_batched_tokens and long_prefill_token_threshold vary.

Each measured window begins at the first observation of running/waiting server requests after the verified restart controller creates a fresh campaign. Metrics every ~2 seconds; hardware every ~5 seconds on both ranks. Ten minutes per arm. Python stack samples for engine and rank0 worker at120s and420s,20s at25Hz, nonblocking. No intrusive GPU kernel tracing in the throughput windows.

Fresh processes and disk-cache lifecycle reset local/external KV state. Filesystem PLE page cache is not flushed; record cold/warm drift as a limitation. Both runtime command lines verified before measurement. Workload controller uses16 frozen arms, up to4 inference slots each, temperature0. Do not change workload sampling or retries.

Preserve full and final-five-minute counter deltas, unfinished request gauges at cutoff, server logs, sampled chunk allocations, hardware and campaign artifacts. Partial-response tokens count toward throughput. Completed useful campaign work must be examined separately. No universal determinism claim.

The controller stops on a failure instead of silently skipping a configuration. Inspect the authoritative process/service state before recovery. Final arm stays running. Analysis and final health checks required before declaring the campaign complete. A ninth drift-repeat is optional and is not part of the eight-arm controller.
