# Dual-Spark inference: NVMe request paging

This branch serves NVIDIA Qwen3.8-Flash-Next-NVFP4 with BF16 KV on two GB10 nodes,
with a 32-request concurrency ceiling and a 262,144-token context limit.
Reservation-based admission queues large requests according to physical cache
capacity. Active requests can park on each node’s NVMe and resume from an
aligned checkpoint with bounded replay.

See [the handover guide](docs/kv-paging-handover.md) for the exact configuration,
validation evidence, limitations, start/stop commands and rollback.

This is an experimental same-process pager. It does not preserve requests
across engine restart or guarantee completed-conversation reuse. The existing
Mia PLE mmap path, TP2+EP, MTP3, 98K draft vocabulary and RoCEnante are retained.

Code is derived from [MiaAI’s dual-Spark setup](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks)
and the local vLLM 0.29.0 branch. The source overlay revision and hashes are in
`paging-manifest.json`; the tested configuration is in `deploy_config.example.json`.
Local runtime files and credentials are gitignored. No credentials are bundled.

The previous release’s description is archived in
[the v0.29 release record](docs/v029-release-readme.md).

## September 12 QSA correctness backport

Both Sparks were updated to `vllm-gb10:v029-roce-qsa55122` with the deterministic
QSA top-k kernel. See the [source pins, build recipe and validation record](docs/qsa-55122/README.md).
