# Attribution and modifications

Based on MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks and MiaAI-Lab's
single-Spark packed PLE and draft-vocabulary tooling, AGPL-3.0-or-later.
Original notices are retained; see LICENSE and provenance.json.

Modified for Jon Taylor's dual-GB10 project, 2026-09-07 through 2026-09-09:
official Qwen FP8/BF16-KV deployment, FP8 packed rows preserving existing PLE
scale/dequantisation handling, checkpoint verification, measurement tools,
service supervision, and configurable paths for the public deployment.

The inference source is maintained in the pinned vllm-gb10 submodule, which
preserves vLLM's Apache-2.0 notices alongside Mia-derived AGPL work.
Model checkpoints and the base container retain their own upstream terms.
