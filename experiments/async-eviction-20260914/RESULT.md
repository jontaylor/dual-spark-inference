# Asynchronous eviction experiment — 14 September 2026

Status: candidate deployed and serving. Pre-pressure and aged disk-restore correctness passed. Source-fence sample share fell in the settled observation, but GPU dips remain and aggregate throughput improvement is NOT established. Candidate left running; no rollback or second restart.

## Exact change

Two existing override implementations changed and one helper was added (22 → 23 overrides). `before-r0/r1.json` and `candidate-r0/r1.json` differ only in override paths and hashes. MTP3, concurrent batching, deterministic numerical kernels, cache policies, sampling and workload retries were retained. `live-r0/r1.json` records Docker arguments/environment with the API secret redacted; `runtime-verified.json` verifies all 23 mounted hashes on both ranks.

Native one-page eviction saves now copy GPU source bytes into a bounded CPU staging region, then release the source fence after copy completion and persistence enqueue. Hashing and filesystem writes continue on the serial disk executor. Public completion/ACK still waits for persistence. Restore paths retain existing checksum and GPU readback verification. Eight buffers hold 217,251,840 bytes per rank (207.19 MiB); buffer exhaustion introduces backpressure. Individual pages are pipelined; multiple pages are not combined into one DMA.

The first actual eviction initializes the region lazily and may incur setup overhead. One service restart was used. `rollback.py` restores the exact previous configuration, but has not been executed.

## Correctness evidence

- Pipeline tests passed: source preservation precedes persistence ACK; bounded-buffer backpressure; copy/write failures propagate and buffers recover.
- Both GPUs passed real 27,156,480-byte page tests: hold persistence, release source fence, overwrite the GPU source, persist, then restore original bytes exactly. Twenty unique source generations exercised eight reusable buffers.
- Live C1/C4/C10 and final serial comparisons: all 15 token-ID/logprob/cache-boundary comparisons passed with background work active. Disk traffic during these checks was zero; they do not test model output after disk pressure.

## Known limits

Cold versus cached numerical parity remains unresolved and is not fixed or claimed by this transport change. The live correctness test does not establish all disk-pressure interleavings. No crash durability/restart-recovery claim. No speedup claim until pressure traffic has been observed and compared. Workload history/cache population changed through the restart, so observational before/after throughput is not a controlled causal estimate.

## Baseline

Read-only pre-change capture: five single-digit GPU episodes over 180 seconds. Rank0 source-fence waiting represented 8.12 / 174.48 successfully sampled main-thread seconds (4.65%); rank1 3.44 / 97.32 (3.53%). Valid metrics covered 150.548 seconds: 175.499 aggregate generation tok/s, 9–12 running requests, 8,146,944,000 stored bytes and 3,638,968,320 loaded bytes across both ranks, zero scheduler preemptions. Summed stack weights are not reconstructed wallclock durations; GPU sensor samples have an internally averaged reporting interval.

Post-change measurements will be appended below.

## Post-pressure correctness

`aged-check-1789408342181342729/summary.json`: replayed the original pre-pressure serial reference using its exact prompt and cache salt after eviction pressure. Serial plus C4 (five comparisons) all matched original tokens and returned logprobs, with 7,327 cached tokens. The serial request restored boundary 7,327 with 101 tokens replay. Job 1312 restored nine disk pages on each rank: 244,408,320 bytes/rank, with explicit GPU readback verification, taking 0.454/0.458 seconds. This is a live aged-checkpoint disk-restore result, in addition to the isolated page tests; it still does not prove every interleaving.

## First pressure window

`post-stalls/summary.json`: valid metrics 179.561 seconds; 103.658 generated tok/s; mean 8.819 running (range 5–10); 28,948,807,680 bytes stored, 380,190,720 loaded; zero scheduler preemptions. GPU low episodes: rank0 nine, rank1 six. Mean GPU power 45.54 / 41.15 W. Source-fence waiting 18.88 / 172.56 successfully sampled main-thread seconds (10.94%).

This is worse raw throughput and more observed dips than the baseline window, but the write rate is approximately three times baseline and concurrency differs. It does not establish a causal regression or speedup. The asynchronous mechanism works, but this capture does not meet the goal of eliminating eviction stalls.

`staging-backpressure.json`: staging thread waited 13.92 sampled seconds for a free buffer. The disk thread spent 22.88 sampled seconds in filesystem stores and 9.80 in persistence (primarily hashing). These overlap main-thread waiting and must not be added as independent wallclock costs.

`fence-evidence.json` matches source-preserved and persistence events by job ID on both ranks. More than 1,000 jobs/rank completed with preservation logged before persistence. These are log timestamps and not exact model-resume timings; the inference fence can still wait on other jobs in the same batch.

`eviction-bursts.json`: approximate engine submission bursts grouped with a 10 ms gap threshold: 96 groups, median four pages, p95 46, maximum 55; 27 groups exceed eight pages. These are approximate log-derived bursts, not scheduler batch IDs. The eight-slot bound allows filesystem latency back onto the source-copy critical path during large bursts. A larger bounded buffer pool is a next candidate, not a tested improvement. No serving buffer-size change was made during observation.

## Below-pressure observation

`below-pressure-stalls/summary.json`: 179.567 seconds; 153.642 generated tok/s; 5–10 running (mean 7.867), no disk traffic or single-digit GPU samples. This is not a performance validation of eviction handling. Earlier server logs contained a short 8.5 tok/s interval with no disk traffic followed by prompt processing and recovery; this interval preceded the stack capture, so its exact cause is unestablished.

## Settled pressure observation and disposition

`settled-pressure-stalls/summary.json`: valid metrics 179.566 seconds; 124.840 generated tok/s; 3–9 running requests (mean 6.894); 12,926,484,480 stored bytes and zero loaded bytes; zero scheduler preemptions. GPU low episodes three on each rank. Mean power 46.28 / 41.87 W. Rank0 source-fence waits 4.88 / 173.48 successfully sampled main-thread seconds = 2.81%, versus baseline 4.65%. The reduction in observed source-fence sample share is encouraging, but differing concurrency, write rates and load traffic prevent a causal aggregate-throughput claim. Single-digit dips remain.

| Metric | Before | Initial pressure | Settled pressure |
|---|---:|---:|---:|
| Valid metrics seconds | 150.548 | 179.561 | 179.566 |
| Aggregate generated tok/s | 175.499 | 103.658 | 124.840 |
| Running requests range | 9–12 | 5–10 | 3–9 |
| Bytes stored, both ranks | 8,146,944,000 | 28,948,807,680 | 12,926,484,480 |
| Bytes loaded, both ranks | 3,638,968,320 | 380,190,720 | 0 |
| Head source-fence sample share | 4.65% | 10.94% | 2.81% |
| Low GPU episodes rank0 / rank1, 180s | 1 / 4 | 9 / 6 | 3 / 3 |

Final log scan on both ranks found no traceback, checksum mismatch, out-of-memory or ERROR entries. All observation processes finished. The candidate remains serving with MTP3 and eight staging buffers/rank. It is not labelled a complete performance fix. The next targeted optimization is to size the bounded staging capacity for observed eviction bursts and validate its memory/performance tradeoff; that change was not deployed in this experiment.
