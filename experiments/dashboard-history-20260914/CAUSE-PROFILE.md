# Live slowdown profile

Nonblocking Python stack sampling, including idle threads, was collected without a serving restart or workload change. Head worker and engine: 45 seconds at 50 Hz. Worker on rank 1: a subsequent 30-second sample at 50 Hz. py-spy 0.4.2 was installed as an operator tool; no inference packages changed. Raw profiles preserve sample errors (head worker 65, engine 359, rank 1 worker 57); this is sampling evidence, not exact GPU timing.

Main-thread samples only:

| Stack category | Head worker (2177 samples) | Rank 1 worker (1433 samples, different interval) |
| --- | ---: | ---: |
| Waiting for output CUDA event | 58.4% | 49.6% |
| Completion checkpoint capture | 10.3% | 18.3% |
| PLE input preparation | 15.9% | 0.3% |
| Short-convolution metadata | 6.6% | 7.0% |
| Other | 8.7% | 24.8% |

Engine main thread: 99.3% of 1888 samples waiting for worker RPC responses. Background monitoring threads are excluded from these percentages. CUDA-event waits include preceding model kernels, collectives and copies; they are not proof of GPU idle time. Stack residency at a synchronization point does not by itself quantify removable overhead because it can wait for preceding work.

A concrete added blocking path is checkpoint-grid capture. Current connector.selective.py lines 371 onward schedules checkpoint requests at computed // 64 * 64 for live generating requests. completion.selective.py avoids recapturing a boundary already attempted, but on each new boundary its capture_completion_states performs per-group GPU block-table .cpu().tolist() calls, captures accepted recurrent states, and copies circular-buffer tensors to CPU bytes. These execute on the model thread. The preserved completion.original.py only invokes its capture body for completion_saves and does not contain the per-grid checkpoint shadow loop. This establishes an added synchronization/copy path in our patch; it does not establish its entire net throughput impact, since it also enables more useful cache reuse.

The current resident restore implementation separately stages GPU-resident pages through CPU memory, checks fingerprints, restores GPU state, and reads back for verification. It avoids filesystem reads but not transfer/verification work.

During the 30-second counter/power interval overlapping the head profile: 134.20 aggregate generated tokens/s, 77.51 W mean sampled combined GPU power, zero filesystem-load byte delta, and 9.46 cumulative restore-job seconds. Thus repeated filesystem reads are not necessary for the observed low-throughput state. Job time can overlap other work. The earlier 30-second sample had filesystem reads; these are different windows.

The earlier readback ABBAABBA live trial found reduced restore seconds/GiB with readback disabled, but no aggregate throughput improvement across its changing workload windows (164.50 tokens/s on, 161.72 off). It does not establish a causal null effect, and it prevents claiming that disabling readback alone is a demonstrated fix.

Conclusion: added synchronous checkpoint capture is a directly observed regression candidate/contributor, and the connector's resident path also incurs host work. The host scheduler is waiting for workers. The full earlier-to-current throughput gap is not causally apportioned: roughly half the worker main-thread samples wait on the GPU output event, and this profile cannot divide that into deterministic kernels, attention/context cost, collectives, or copies. A controlled GPU timeline or equivalent phase-timing A/B remains needed to name a single dominant cause. Do not claim the whole gap is disk I/O or checkpoint overhead.

Targeted correction to evaluate: preserve the 64-token arithmetic/checkpoint semantics while batching capture, keeping snapshots on GPU where feasible, and removing per-request/per-group blocking host round trips. Do not remove the arithmetic grid or serialize requests to hide the issue.
