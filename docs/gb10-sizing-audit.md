# GB10 buffer, batch and launch sizing — initial verification

The serving setup contains several measured GB10 optimisations and correct alignment safeguards. It does not have a complete, measured hardware-specific sizing policy for every hot path. The most concrete outstanding checks are prefill remainder allocation, three-request BF16 plan coverage, inherited sparse-attention tile/split profiles, and CPU thread placement across unequal cache domains.

This audit queried hardware attributes on both nodes without creating a current CUDA context, inspected source and configuration, and reused an earlier C1/65K capture. It did not change serving code/configuration or run inference benchmarks. Launch measurements below belong to the earlier capture, not a new v0.29/C4 trace.

## Hardware actually reported

Both nodes returned identical queried GPU attributes and CPU cache topology.
The full host-local records, `head-hardware.json` and `worker-hardware.json`,
remain outside this public repository.

| Resource | Reported value |
|---|---:|
| GPU SMs (execution clusters) | 48 |
| Threads per warp | 32 |
| Maximum resident threads per SM | 1,536 |
| Maximum resident blocks per SM | 24, subject to other resource limits |
| Registers per SM | 65,536 × 32-bit |
| Shared memory per SM | 100 KiB maximum |
| GPU L2 | 24 MiB |
| CPU L1 data | 64 KiB per core; 64-byte cache lines |
| CPU L1 instructions | 64 KiB per core |
| CPU L2 | Ten private 512 KiB caches + ten private 2 MiB caches |
| CPU L3 | 8 MiB shared by CPUs 0–9; 16 MiB shared by CPUs 10–19 |


CPU core type and L3 cluster are separate: CPUs 0–4 and 10–14 are A725s
with 512 KiB private L2; CPUs 5–9 and 15–19 are X925s with 2 MiB private L2.
The 0–9 cluster shares 8 MiB L3, and the 10–19 cluster shares 16 MiB L3.
Both nodes confirm this mapping through MIDR and cache sysfs. The current
frequency driver reports 2.808 GHz for A725 and 3.9 GHz for X925 on both
clusters. This corrects earlier wording that implied a whole cluster had the
larger private L2. The user's reported 4 GHz boost and possible extra
memory/SLC connectivity on cluster 1 remain unverified by these interfaces.

CPU L2 totals 25 MiB, but is not one uniformly accessible cache. The two CPU L3 domains likewise should not be treated as a single 24 MiB pool available to every CPU or GPU load. These interfaces do not establish all SLC/coherence routing policies or expose every GPU data/instruction cache; no unreported cache sizes are assumed.

## Verified and unresolved sizing choices

| Area | Verified | Remaining question |
|---|---|---|
| Decode graphs | Explicit capture sizes 4, 8, 12, 16 match C1–C4 target verification with MTP3; C4 does not require padding up to a 32-position graph. | Internal operators can still pad dimensions or expert rows. |
| BF16 matrix kernels | 40 measured GB10 plans; packed row-major/precision/shape checks before dispatch. Plans preserved across migration. | No M=12 plans; only some weight shapes have M=16 plans. Standard `F.linear` is the intentional fallback, whose relative performance needs measurement for these shapes. |
| Prefill | Uses the resolved 1,600-token recurrent-state alignment. Total ceiling 8,192; fair per-request ceiling at four prefills is 2,048, commonly rounded down to 1,600. | Uniform sharing commonly leaves 1,792 scheduling positions unassigned. An extra 1,600-token quantum could potentially go to a rotating request, subject to checkpoint/budget constraints. |
| Sparse-attention scoring | 64-column tiles, two warps, two stages; grid scales with rows and cache-table capacity. | Source profile explicitly says tuned on GB300. GB10-specific best tile/stage/warp settings are not established. |
| Sparse attention proper | Splits long work to provide more GPU blocks; caps split count to avoid empty splits; uneven splits use dynamic loop bounds. | Split/warp/tile profiles also say tuned on GB300. Need measured occupancy, traffic and step time at actual GB10 shapes. |
| MoE | Compact expert GEMMs use 16-row tiles, 128-element reduction chunks, four warps and two stages; supported shapes validated. Sparse activation follows routed rows. | Launched blocks include early exits for nonlocal/duplicate expert ownership. Raw grid size alone overstates useful work; route balance and active tile density matter. |
| CPU PLE rows | Cache values aligned to 64 bytes. Packed FP8 rows remain 160 bytes, preserving Mia's format and global scale. | Need hot-set/reuse and cache-miss measurements to choose software-cache size; total table size is not the hardware-cache working set. |
| CPU/GPU flags | Producer/consumer fields separated by 64-byte offsets, using release/acquire publication. | CPU cache-line separation does not prove absence of all GPU/coherence-sector contention. |
| CPU gather threads | Eight gather threads, static OpenMP partitioning, passive OpenMP wait policy. | All observed head service threads are eligible for CPUs 0–19; no explicit placement in one cache domain or isolation from engine threads. |

Relevant source references in the selected checkout:

- `dual-spark-inference-v029/launch_rank.py:134`: decode capture sizes.
- `vllm-gb10-v029-pr/vllm/v1/core/sched/gb10_prefill.py:7`: fair prefill ceiling.
- `vllm-gb10-v029-pr/vllm/v1/core/sched/scheduler.py:396`: alignment and checkpoint clipping.
- `vllm-gb10-v029-pr/gb10/bf16_plans.json`: measured plans.
- `vllm-gb10-v029-pr/vllm/models/qwen4_exp/nvidia/low_latency_gemm.py:139`: dispatch/fallback.
- `vllm-gb10-v029-pr/vllm/models/qwen4_exp/nvidia/ops/qsa.py:640`: scoring geometry.
- `vllm-gb10-v029-pr/vllm/models/qwen4_exp/nvidia/ops/qsa.py:857`: sparse-attention profiles.
- `vllm-gb10-v029-pr/vllm/model_executor/layers/fused_moe/experts/gb10_moe_gemm.py:24`: expert ownership/tiling.
- `vllm-gb10-v029-pr/gb10/ple_gather.c`: row-cache allocation and gathering.
- `vllm-gb10-v029-pr/gb10/mapped_wait.cu`: publication/wait fields.

## Interpreting the prefill remainder

8,192 is a scheduling ceiling. It is not a statement that every GPU kernel performs 8,192 tokens of work. Prefill is not captured by the full-decode-only graphs. Therefore 6,400 scheduled tokens is 78.125% of this token budget, not evidence of 21.875% GPU idleness.

There is nevertheless a concrete policy opportunity: for four long non-final prefills, 3,200 + 1,600 + 1,600 + 1,600 = 8,000. Rotating the larger share could retain fairness across iterations. The current uniform per-request cap does not do that redistribution. Whether this improves throughput depends on per-step cost, interference with decoding, state-copy lifetime and peak memory. No 25% speed gain follows from the 25% increase in token positions.

Simply setting the ceiling to 6,400 would make the budget look full without itself making the same work faster. Increasing the ceiling until every request gets a larger aligned chunk changes peak memory and responsiveness. Both require evidence, not a preferred round number.

## GPU tail effects are real, but residency matters

The proposed 50 equal blocks on 48 SMs is a useful example if only one block can reside on each SM. It produces an almost-full first wave and a two-block tail. Under those simplified assumptions, the two-wave average is about 50/(2×48) = 52.1% occupied block slots. Actual scheduling dispatches blocks as resources free; multiple blocks can reside on an SM and their runtimes need not be equal.

The relevant capacity depends on threads, registers, shared memory, allocation granularity and other kernel constraints. A grid does not generally need to be a multiple of 48. Arbitrarily adding dummy blocks does not create useful parallelism. Splitting useful work into smaller blocks may help, but can also duplicate reads or require a reduction.

The saved earlier C1/65K launch inventory (`old-c1-65k-launches.json`) supplies examples:

- Compact expert kernels launched 20×40 = 800 blocks, at 128 threads/block. Different specializations used 72 or 114 registers/thread and about 10–19 KiB shared memory. Many blocks can exit early due to routing/ownership, so useful block density needs separate analysis.
- QSA scoring launched 4×1025 = 4,100 blocks, at 64 threads/block, 79 registers/thread and 20,992 bytes shared memory. Blocks beyond visible context return early. A small 48-versus-50 tail is not a sufficient description of this workload.
- QSA sparse attention launched 4×1×64 = 256 blocks, at 128 threads/block, 108 registers/thread and 25,216 bytes shared memory. The shared-memory ceiling alone permits at most four such blocks per SM; that is a resource bound, not a measured achieved occupancy.
- Fused GDN MTP launched 1×24 = 24 blocks at C1: fewer blocks than SMs, with 256 threads, 72 registers/thread and 45,120 shared-memory bytes per block. The corresponding source grid is `(num_requests, num_value_heads)`, so four requests with 24 local value heads give 96 blocks. These captured calls total about 0.822 ms per target iteration against a 64.0 ms mean target-start interval, about 1.3%. Splitting C1 work further is a real candidate, but dependencies, traffic and this limited share constrain the possible overall gain.
- Collective and readiness kernels can legitimately launch few blocks because their job is communication or waiting. Counting their idle SMs as a compute-tiling bug would be misleading.

Kernel launch resource values are from the prior capture. They cannot establish current v0.29 register allocation or C4 occupancy without a current compiled-kernel check/trace.

## Cache-fit checks need the reused working set

The configured 256 MiB PLE software-cache budget becomes 1,048,576 slots after power-of-two rounding: 160 MiB of rows plus 8 MiB of tags, about 168 MiB plus small metadata. It is a software cache of mmap rows, not an attempt to fit the entire allocation into CPU L2 or L3. Its benefit depends on avoiding repeated random table/page accesses.

For a 64-byte-aligned base, 160-byte rows alternate between offsets 0 and 32 modulo 64. Each random row intersects three cache lines. Padding a row to 192 bytes would still intersect three lines while increasing stored size by 20%; alignment by itself does not establish a benefit. Consecutive packed rows can share cache lines.

Tiling aims to keep data available until its next reuse, considering concurrent blocks and other users of that cache. Capacity alone is insufficient: associativity, strides, reuse distance, coalescing and contention matter. Instruction caches hold executable instructions, so data-buffer size is not their direct tuning parameter; excessive unrolling or code specialization can affect them indirectly.

## Basic measurements to do before broader changes

1. Compare current prefill allocation with redistribution of the leftover aligned quantum, using the same long prompts and recording both prefill time and decode interference. Protect existing cache/checkpoint invariants.
2. Check C3 and partial-batch BF16 shapes against standard kernels, extending the measured plan set only where a win exists. Current C1/C2/C4 benchmarks do not cover every transient batch size.
3. Measure QSA tile/split/warp choices at C1–C4 and the real cache lengths. Record bytes moved, shared memory/register pressure, spills, achieved occupancy and elapsed time together.
4. Compare CPU gather placement on fast cores in each L3 cluster, across fast-core groups, and with suitable thread counts, while leaving engine/network threads enough CPU capacity. Read-only inspection establishes that placement is currently unrestricted; it does not prove migration or contention.

These are sizing and placement checks. A clean pass means measured suitability for the selected shapes/workload, not a claim that every buffer must exactly fill a cache or every launch must exactly fill a wave.
