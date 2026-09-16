Activated and healthy

Both ranks: PLE row-cache budget 256 → 1024 MiB; gather threads 8 → 16; NCCL_IB_HCA =rocep1s0f0 → =rocep1s0f0,roceP2p1s0f0. Exactly these three configuration fields changed. Actual Docker environments verified on both ranks; live arguments, environments and mounts recorded in live-r*.json.

Health HTTP200. Inference smoke passed, returning ready (28 completion tokens, 1.52s wall time). No ERROR or Traceback found in captured startup logs. RoCEnante remains enabled. Workload task notified of readiness; resumption confirmation pending. No performance or full-determinism conclusion from this smoke test. Cache budget is 1GiB; power-of-two layout expected to allocate672MiB for4194304 slots per rank, not yet confirmed by periodic cache statistics.
