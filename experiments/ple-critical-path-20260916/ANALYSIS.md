**PLE throughput and critical-path review — 16 September 2026.**

This is a read-only review of serving code and existing measurements. No server
restart, backend switch, affinity change, new load, or new GPU instrumentation
was performed. The numerical breakdown is reproducible with `analyze.py` and is
saved in `evidence.json`. It describes historical captures, not current traffic
or a causal comparison with the original external-worker implementation.

The objective is maximum useful serving throughput at an acceptable request
latency and fairness. PLE reader duration is only a component measurement. An
improvement matters when it removes exposed work, reduces contention, or lets
the GPU sustain more useful work. Speeding a read that already finishes before
its consumer may leave throughput unchanged.

**The deployed dependency chain explains the apparent contradictions.**
The model has one PLE embedding at transformer layer 2 (zero-based index 1).
Its hashes use the input token and up to two preceding tokens, with eight heads
per n-gram length: 16 rows per input token. With three MTP draft tokens, a
normal target verification step has four input positions and 64 row references
per request. Twelve requests produce 768 references; 32 produce 2,048, before
deduplication and any graph padding. Each FP8 row is 160 bytes: a normal
32-request batch produces 320 KiB of raw embeddings per rank. The packed table
is 51,200,245,760 bytes (about 47.7 GiB) per local replica. An 8,192-token prefill
can require 131,072 references and 20 MiB of output before accounting for reuse.

For FULL graph execution, the current sequence is:

1. Prepare input tokens and n-gram histories on the GPU.
2. Hash row IDs on the GPU, then synchronize a CPU-visible completion event.
3. On the GPU worker's main CPU thread, deduplicate, consult the row cache and
   submit missing reads. All unique misses are submitted before completion waits.
4. Launch the full model graph; the first transformer layer can now execute.
5. On that same CPU thread, finish reads, copy into mapped output, update the
   cache, and publish one batch completion flag.
6. At layer 2, the GPU checks that flag, then reads/dequantizes the embeddings
   and performs the PLE projections and combination with hidden states.

For eager/piecewise execution the entire gather completes before model launch.
The synchronous path includes prefill and other non-FULL shapes; it should not
be labelled exclusively prefill. Deferring completion until after a synchronous
model call without an independent producer can deadlock at the consumer wait.

There are consequently four places to account for PLE: CPU/GPU synchronization
and preparation before graph launch; exposed waiting at the layer; GPU memory
access/dequantization/projections after readiness; and resource contention with
the rest of inference. The existing wait-kernel metric measures only the second.
The hash-to-ready interval includes intervening useful GPU work and launch delays.
Neither metric measures GPU L2 residency or the complete end-to-end PLE penalty.

For a simple overlap model, let A be PLE startup time before useful model work
can launch, L the remaining time to a ready result after launch, and E the useful
GPU work before the PLE consumer. Exposed startup/wait cost is approximately
`A + max(0, L - E)`, excluding access costs and contention. This is a dependency
model, not an estimator from the existing counters. If L is already less than E,
reducing L has no direct step-time benefit. Increasing A to decrease L can regress
the step. Across tensor-parallel ranks, readiness and communication dependencies
must be followed together; rank times cannot simply be added.

**The existing measurements substantially narrow the opportunity.**
Recomputing the earlier 20-minute fixed-policy capture gives:

| Measurement | Rank 0 | Rank 1 |
|---|---:|---:|
| FULL graph batches | 5,147 | 5,157 |
| Reader duration, median | 1.720 ms | 1.881 ms |
| Native submission, median | 0.0344 ms | 0.0357 ms |
| GPU wait body, mean | 5.21 us | 6.51 us |
| GPU wait body, p95 | 0.384 us | 0.512 us |
| GPU wait body, maximum | 3.269 ms | 1.748 ms |
| Batches with at least one wait-loop poll | 1.90% | 2.37% |
| CPU step-period proxy, median | 168.032 ms | 168.029 ms |
| Synchronous batches | 733 | 733 |
| Synchronous reader duration, mean | 2.690 ms | 2.736 ms |
| Synchronous reader duration, p95 | 10.808 ms | 10.993 ms |
| Synchronous reader duration, maximum | 57.369 ms | 57.175 ms |

There are 5,146 aligned FULL batches with wait data on both ranks. The per-batch
rank-maximum wait averages 10.41 us and sums to 53.55 ms. That maximum is only a
descriptive aggregation, not an estimate of tensor-parallel critical-path delay.
The direct wait was a small average cost in this workload; rare tails remained.
This does not bound the total cost of hashing, host synchronization, launch delay,
GPU embedding access, or contention. The capture predates the engine-unpin
change and is not an affinity comparison.

Synchronous native reader duration totals about 1.97/2.01 seconds per rank in
that 20-minute capture. This is another indication that the measured reader
alone did not consume a large fraction of that workload's wall time. It is not
a throughput bound: the preceding synchronization, later GPU PLE operations,
rank skew and indirect memory effects are not included.

The later ABBA capture had much heavier synchronous work. Even the previous
reader policy recorded synchronous gathers exceeding 90 ms. It confirms that
different workload phases expose different costs; it does not prove how much
throughput an alternative could recover. Eager waits are near zero precisely
because the CPU has already waited before launching the model.

The latest page-prefetch candidate won isolated warm/cold tests but moved a
synchronous `process_madvise` call onto the graph-launch path. Submission grew
from roughly 0.033 ms to 0.5–0.9 ms. Matched live hash-to-ready intervals regressed
by about 0.44–0.63 ms. The whole-step comparison had only one complete matching
cycle, so it establishes neither overall superiority nor an overall slowdown.
Rejecting the candidate was justified; calling the test a proven end-to-end
regression would overstate the evidence.

Earlier SQPOLL-vs-normal submission tests demonstrated readiness-path gains,
but whole-step gains were unresolved. They compared two replacement policies,
not original external PLE against the final in-process design. No reviewed
measurement establishes which complete implementation maximizes serving throughput.
This distinction should have governed the earlier optimization work.

**The earliest possible start depends on the kind of work.**

| Work | When the required row IDs can be known | Useful lead time |
|---|---|---|
| Prompt / later prompt chunk | After tokenization and identification of the uncached token span, using the preceding two tokens | Queue time and execution of earlier chunks or other requests |
| Next ordinary decode token | After the preceding model output is sampled | Remaining postprocessing, scheduling and work from independent requests |
| MTP verification tokens | As the sampled token and each draft token become available with its n-gram history | Remaining drafting and subsequent host preparation; not an entire preceding target forward |
| Unknown future generated tokens | Not exactly known yet | Only speculative prefetch, with wasted work on rejected guesses |

The current connector starts from the next batch's prepared GPU inputs. Starting
at the producer of those IDs can be earlier. Prompt prefetch can avoid a GPU hash
round trip entirely if exact CPU hashing and boundary semantics are preserved.
Decode should not force an extra GPU-to-CPU synchronization just to prefetch.
Prefetched rows can be keyed by row ID; final output packing still follows the
eventual batch order. Prompt prefix hits need no PLE for tokens whose model work
is skipped. MTP rejection, EOS handling, resumed requests and chunk boundaries
must retain exact semantics.

**The implementation choices are independent axes, not one winner-takes-all choice.**

| Option | What it could improve | What could make it lose / required work |
|---|---|---|
| In-process synchronous native gather for guaranteed-resident rows | Very low overhead for small hot batches; no read syscall needed for an owned RAM cache hit | A software-cache miss may still be hot in the OS cache, but arbitrary mmap loads can fault and block the launch thread. Dispatch must be based on reliable state, not an expensive per-row residency syscall. |
| In-process asynchronous producer | Keeps hashing/input staging/I/O/gather away from the launch thread and can support overlap for eager as well as FULL execution | Needs a persistent native producer, explicit buffer lifetime, generation/error state, and bounded queues. Wakeup and contention costs remain. More threads are not intrinsically better. |
| External worker with shared buffers and ZeroMQ or a shared-memory queue | Separates progress from the model thread; can overlap the entire preparation and read phase | Control handoff and wakeup can dominate tiny hot batches. ZeroMQ can carry a small batch descriptor while embeddings stay in shared memory. Full-path measurement must include handoff, copies and GPU consumption. |
| Bounded prompt prefetch | Uses exact known tokens to move disk/page faults before scheduling; next-chunk prefetch can overlap the current chunk | Unlimited prefetch competes with KV paging and evicts useful memory. Needs a memory budget, priorities, cancellation and ownership until use. |
| Readiness-aware scheduler | Excludes genuinely late work from a batch so other ready requests can execute | Needs per-request/chunk readiness across both ranks and a lease on the prepared data. Smaller batches can lose throughput. Include age/fairness and a bounded wait-vs-run decision rather than always skipping misses. |
| Multiple independent microbatches / split graphs | GPU work from batch B may overlap PLE for A | Current single output/flag/reader ownership cannot do this. Needs multiple buffer slots, graph/state lifetime changes, resource-aware stream execution and consistent TP/EP collective order. Existing DBO compatibility is not a ready-made solution. |
| Better caching / GPU-addressable hot rows | Avoids disk/kernel work and possibly an intermediate copy; exact row and n-gram reuse is available | Whole-table residency costs about 47.7 GiB per node. GB10 CPU and GPU share physical RAM; extra cache can reduce KV capacity/concurrency. Benchmark cache miss rate, memory pressure and throughput together. |
| Earlier GPU access / embedding-only projections | Read or project completed embeddings before their hidden-state consumer; key/value projections depend on embeddings alone | Requires a safe dependency branch and memory ownership. Extra GPU work competes for compute/bandwidth. No assumption that prefetch guarantees lasting L2 residency. |
| Progressive row/chunk completion | Starts copies or processing before the slowest row finishes | Current code copies only after every read succeeds, then publishes one batch flag. Benefit needs independent downstream work or chunked consumers; a final batch barrier alone preserves head-of-line delay. Requires generation-safe partial publication and errors. |
| Storage / data layout / transport changes | Page coalescing, registered buffers/files, appropriate queue depth, direct reads into staging, SSD power policy, possibly specialized I/O | Tiny cached reads can be faster as loads. Cold 160-byte accesses may fetch 4-KiB pages. Direct I/O, SPDK and GPU Direct Storage need platform validation and may lose cache benefits; no assumed GB10 support or free gain. |
| Table sharing across hosts or further compression | Reduces duplicated capacity or physical bytes read | Network latency and additional dependencies can exceed saved local I/O; lossy formats/model changes need a separate quality evaluation. Neither is an immediate latency fix. |

The packed FP8 format, a bounded software row cache, per-batch deduplication,
mapped output, all-misses-before-wait submission, FULL-graph overlap and SSD
latency constraint already exist. They should not be counted as new proposals.
The software miss counter does not distinguish OS-page-cache hits from physical
disk misses. Sixty-four application workers cannot make 64 cached memory loads
inherently faster; cold-read parallelism should be sized to useful queue depth.
Large prefills also need bounded I/O so they do not starve decode or KV traffic.

The GPU does not automatically pick another vLLM request when this batch reaches
its PLE wait. Requests are fused into the batch; the completion flag covers all
of them and later tensor-parallel work requires consistent progress. One late
row or rank can hold the batch. Independent GPU work can overlap only if the
runner has actually queued it with compatible resources and dependencies. A
polling wait can even appear as GPU activity without doing useful model work.

Data merely being in reclaimable page cache is not a readiness guarantee. A
scheduler must know that prepared data is retained until consumption, and both
ranks must agree on the batch. RAM-ready also does not mean L2-resident. The
practical contract is correct, GPU-accessible data with the required lifetime
and memory ordering, plus measured cost of consuming it.

**Measurement should establish the size of the opportunity before selecting another backend.**

1. Account for one serving step from input availability through real accepted
   token output: PLE CPU staging/synchronization, read submission, GPU launch,
   entry/exit of the PLE wait, first embedding consumer, forward/draft end and
   accepted-token count. Keep FULL decode, synchronous decode, prefill and mixed
   batches separate. Link both ranks by step ID. Measure physical disk misses
   separately from software-cache misses and account for KV I/O and communication.
2. Use the existing traces immediately; a short sampled GPU timeline can resolve
   remaining launch/consumption attribution. It can perturb execution, so measure
   overhead and do not infer GPU idleness solely from the wait body or utilization.
   The current monitor gives descriptive groups, not exact context or causal ranking.
3. Establish an opportunity ceiling with representative captured steps and exact
   precomputed PLE outputs. Compare the same model computation with those correct
   outputs ready in advance. Include their memory-access cost. This is a controlled
   replay experiment, not a production shortcut with zero/random embeddings. A
   replay test requires the appropriate GPU allocation; the serving model cannot
   safely share an unplanned second large CUDA workload.
4. Install a common backend contract once: identical hashing, outputs, cache
   accounting, generation/lifetime rules and instrumentation, with explicit runtime
   routing between the original full external path and candidate paths. The current
   runtime control only chooses SQPOLL vs rejected prefetch; it does not provide
   original-vs-in-process switching. New loaded code requires a planned deployment,
   but subsequent comparisons need not restart the model.
5. For reader mechanisms, use randomized repeated in-process time blocks with both
   ranks aligned and equivalent cache state. Report confidence intervals over whole
   blocks/cycles, with exact batch/context/prefill/speculation mix. Shared caches
   reduce drift for read-policy tests but confound cache-policy comparisons. Avoid
   concurrent shadow I/O: it can warm the file cache and contaminate the control.
6. For scheduler/cache changes, preserve a controlled offered workload, allow for
   queue/cache carryover, and compare longer repeated blocks or deterministic
   request-trace replays. Fixed per-step inputs isolate execution costs but cannot
   prove a scheduler gain because scheduling is what that change alters.
7. Judge useful accepted output tokens per second under sufficient demand and a
   fixed prompt-work mix, alongside prompt progress, time to first token, per-request
   time per output token, tails, fairness, memory pressure and errors. A smaller
   batch can improve individual token latency while lowering aggregate throughput.
   Requests crossing a mode change have mixed exposure; completed-request means
   cannot be assigned wholly to the mode at completion. MTP output-burst timing is
   not one event per generated token. End-to-end quality remains a separate check.

The recommended order is to quantify unhidden launch/consumption costs, then
test an independent in-process producer and bounded exact prompt prefetch. Keep
the original external worker as a full baseline and retain a cheap resident-row
path. Add scheduler readiness only if measured late work materially blocks
otherwise useful batches. Microbatch pipelines, table placement and specialized
storage deserve investment only if the measured remaining burden warrants it.
If correct ready-in-advance replay barely improves whole-step throughput, further
PLE optimization should give way to the actual compute, memory, communication,
KV-paging or scheduling bottleneck.

Evidence and source anchors (paths relative to the deployment root unless stated):

- `experiments/ple-latency-campaign-20260916/FINAL_REPORT.md`
- `results/ple-latency-campaign-20260916/final-stability/rank{0,1}-steps.csv`
- `results/ple-resident-20260916/live-ab/rank{0,1}-steps.csv`
- `results/ple-resident-20260916/selection.json`
- `experiments/ple-latency-campaign-20260916/variants/17-fixed-policy/in_process.py`
- `experiments/ple-latency-campaign-20260916/variants/06-overlap/model_runner.py`
- `experiments/ple-latency-campaign-20260916/variants/12-gpu-gate-timing/mapped_wait.cu`
- `experiments/ple-resident-20260916/ple_batch_reader.c`
- `experiments/targeted-determinism-20260913/model.final.py`
- `experiments/ple-in-process-20260916/ple_layer.py`
- `/home/jon/vllm-gb10-kv-paging/vllm/models/qwen4_exp/nvidia/model_state.py`
- `/home/jon/vllm-gb10-kv-paging/vllm/v1/ple_offload/{connector,worker}.py`
