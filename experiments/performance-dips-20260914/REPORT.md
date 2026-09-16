# Live throughput dips

The server is healthy, but mixed long-prompt traffic causes real decode slowdowns. The reported 5–8 tokens/second values occur in the server's aggregate generation metric, so they are not solely a client averaging artifact.

## Evidence

A 15-minute log sample contains 90 ten-second throughput windows. KV accounting snapshots identify requests still prefilling independently of the displayed prompt-throughput counter.

| Window classification | Windows | Median aggregate generation tokens/s | Range |
|---|---:|---:|---:|
| No prefill observed | 29 | 129.5 | 67.0–172.8 |
| Some prefill observed | 61 | 8.2 | 4.0–113.9 |
| Average of at least two prefills | 6 | 5.95 | 5.2–7.9 |

These groups reflect the changing live workload, not a controlled benchmark; some windows include transitions. The last group is a subset of the second.

At 01:50:21 UTC the server generated142.3 tokens/s with eight running requests and no observed prefill. At 01:50:31 it generated130.1 tokens/s. During01:50:41–01:51:31 it fell to5.2–7.9 tokens/s as multiple long prefills ran.

The low-speed accounting records show three prompts of66,761,79,124 and72,913 tokens each advancing in1,600-token chunks. Five existing decoding requests share those steps and advance by only a few tokens between roughly three-second snapshots. This directly demonstrates that decoding waits behind substantial prompt computation, even though the requests are all marked running.

Speculation remains active. For example,01:50:41 reports86.3% acceptance while generation throughput is6.0 tokens/s. Reduced acceptance contributes in other windows, but it cannot explain the whole collapse.

The current fair-prefill helper divides the8192-token batch budget among prefilling requests; it does not impose a smaller limit when decoding requests are present. Mamba cache boundaries commonly reduce chunks to1600 tokens. With three prefills, that can place4800 prompt tokens alongside only a few verification tokens per decoding request.

## Other checks

- No scheduler preemptions in the live cumulative counter; no waiting requests at the sampled endpoints.
- KV disk-load byte counter remained zero. Disk stores continued, so storage overhead cannot be excluded, but the evidence does not point to repeated KV disk restores as the primary cause.
- Both hosts had zero swap-in/swap-out during the short vmstat samples. Previously allocated swap remains substantial; this is not proof against all memory-pressure episodes.
- A head GPU sample showed96% utilization,71°C and2509MHz. No detailed kernel or CPU-wait profile was taken, so the cost within prefill (attention, GEMM, PLE fetching) has not been apportioned.
- Prompt-throughput logs can show zero while accounting snapshots show prompt positions advancing. They must not be read as evidence that no prefill is occurring.

## Recommended next change

Test a decode-aware prefill budget: smaller chunks, initially128 or256 tokens on the existing64-token arithmetic grid, whenever decoding requests are active, with larger chunks retained for prefill-only batches. Apply a total prefill budget as well as a per-request limit so multiple prefills cannot collectively monopolize a step. This preserves request concurrency and can retain the fixed reduction schedules.

This trades some prompt-processing efficiency/latency for smoother decoding. Measure inter-token latency, aggregate generation throughput and prefill completion time together, then repeat exact token/score comparisons under changing chunk sizes. The current full-serving determinism tests do not by themselves validate a new scheduling policy.

No serving source, configuration, workload setting or running process was changed during this inspection. Raw evidence: server.log, metrics-start.txt, metrics-end.txt and analysis.json; analyze.py reproduces the window classification.
