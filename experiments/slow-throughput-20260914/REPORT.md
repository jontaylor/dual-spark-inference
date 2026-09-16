# Current throughput diagnosis

Read-only investigation; MTP3, workload configuration, and serving cache policy
were not changed. No restart or synthetic inference benchmark was performed.

## Findings

1. Native matching is coarse: physical block size 1920, no explicit
   prefix_match_unit, so prefix-cacheable group GCD resolves matching to 1920.
   Native speculative lookup drops an additional aligned block. Removing exact
   completion snapshots exposed this existing native replay behavior.
   Across 125 recorded responses created after 12:19:20 UTC,
   uncached input summed to 490,180 tokens and output to
   92,214. Median uncached input was 3,711
   tokens versus median output 410. Uncached input includes
   genuinely new content; these totals are not all attributable to alignment.
   The source and earlier controlled follow-up probe establish the replay tax.

2. Offered concurrency has fallen. At 12:35:20 UTC five of eight campaign arms
   were complete; three remained running. The server had no waiting requests in
   any of 90 sampled ten-second windows. Decode-only windows averaged 73.2 t/s
   with 2.58 running requests, 105.0 t/s with 4.28, and 140.0 t/s with 6.39.
   These are observational windows, not matched isolated kernel benchmarks.

3. Prefill interactions reduce visible output further. Windows containing a
   prefilling request averaged 83.6 output t/s with 5.21 running requests;
   decode-only windows averaged 107.6 with 4.49. Context lengths, acceptance,
   and workload composition also vary, so this is not an isolated penalty.

4. Native eviction writes are wasteful but not the strongest measured cause.
   Rank 0 recorded 421 disk writes, 10.65 GiB, over the sampled 15 minutes.
   Their summed transfer duration was 14.93 seconds; largest transfer 0.163 s.
   This sum is not the whole two-rank critical-path cost. There were no disk
   restore reads. Periodic Mamba checkpoint retention is currently 1920,
   equal to the physical block size: the native mask therefore retains every
   eligible checkpoint. This explains cache pollution by intermediate states.

## Corrective direction

Retain useful native GPU state at finer boundaries so a follow-up avoids
multi-thousand-token replay, while sharing attention pages and retaining
completion/junction state instead of every intermediate recurrent checkpoint.
The installed vLLM has prefix_match_unit (CLI --prefix-match-unit) and semantic
retention (prefix_cache_retention_interval=0). These are concrete native
mechanisms to evaluate, not verified fixes merely by setting their flags.
The deterministic 64-token arithmetic grid, speculative accepted-state handling,
QSA compression state, partial-tail COW, and disk key compatibility still need
validation before deployment. Larger active batches are needed for historical
200+ aggregate t/s comparisons; completing benchmark arms should not be confused
with a scheduler concurrency cap. Existing workload was left unchanged.

## Profiling limitation

A nonblocking py-spy capture requested for 45 seconds did not terminate and
produced no usable artifact. It was stopped after roughly three minutes. No
stack-sampling conclusion is claimed. Inference continued throughout. GPU
utilisation was 95% at the initial sample, which does not measure kernel
occupancy or establish an isolated model-computation bottleneck.

Raw evidence: response-analysis.json, window-analysis.json, decode-bands.json,
disk-events.json, server.log.
