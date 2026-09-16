# Targeted HyperConnection GEMM candidate

Replaces the six projection call sites in `GatedResidual.mix` and `combine_and_mix` with vLLM `matmul_persistent` using the existing unquantized, bias-free replicated weights. Preserves checkpoint loading, padding, RMSNorm, activation, gate/combine kernels, and all attention backends. Does not set `VLLM_BATCH_INVARIANT` or bypass its unsupported-GDN guard.

The original first-mixer divergence was independently reproduced using checkpoint weights and cuBLAS, matching the captured serving tensors exactly. See ../gdn-trace-20260913/REPORT.md for the diagnosis and primitive controls.

Predeployment validation: `class_replay.py` loads the actual patched `GatedResidual` class and runs its `mix` method on captured serial and mixed-forward inputs with the checkpoint weights. Block input and injection outputs match bit-for-bit for all three matching mixed-batch slices. Evidence: `hc-class-replay.json`, `class-replay.log`.

Runtime override: `models/qwen4_exp/nvidia/hyperconnection.py`, hash-verified on both nodes. No launcher edits. Baseline configurations saved per-node in `config.before.json`; candidate in `config.candidate.json`.

Full-server startup passed. Original replay: 10 successful responses, one serial output, four concurrent outputs, none matching serial; every concurrent sequence first differs at token 3. The targeted patch is therefore insufficient. Fixed-seed repeat also completed: all 20 requests successful, all 12 serial bodies identical, the same set of four distinct concurrent bodies across repeats. Isolated mixer equality does not establish full-model determinism. A passing response test would cover the tested workload only; additional model operations and batch shapes can still need deterministic implementations.

Final state: the unsuccessful targeted candidate was removed from both deployment configurations. Recovery of the prior serving configuration is running in the background; API readiness has not been re-verified after this last restart. Speculation remains off. No production serialization gate or successful concurrent-serving fix is installed.
