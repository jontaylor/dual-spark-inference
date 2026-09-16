# In-process PLE source snapshot

This patch preserves the remaining experimental changes in the working server
checkout at publication time, including native build integration and reader
regression tests. Base revision: `01b52a02e5e339a938715639d3cfa8e5def06dc1` in jontaylor/vllm-gb10.

It is an archived candidate, not the external-only production selection, and
not necessarily the exact variant used in the final comparison. The comparison
folder records its separately frozen sources and hashes.

Apply `in-process-source.patch` to a clean checkout of the base revision.
`git apply --check` passed against the exact base file contents. The source
checkout was not modified or committed during this archive operation. GPU tests
were not rerun alongside the serving model.

Included paths:

- `gb10/build.sh`
- `gb10/build_mapped.py`
- `gb10/mapped_wait.cu`
- `vllm/model_executor/layers/ple_offload_layer.py`
- `vllm/models/qwen4_exp/nvidia/ple_layer.py`
- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu_worker.py`
- `gb10/PLE_IN_PROCESS.md`
- `gb10/ple_batch_reader.c`
- `gb10/ple_hash.cu`
- `tests/gb10/test_ple_batch_reader.py`
- `vllm/v1/ple_offload/in_process.py`
