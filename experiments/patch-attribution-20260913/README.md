# Attribution of concurrency divergence to local patches

This comparison holds the NVIDIA NVFP4 checkpoint revision, archived request, temperature zero and TP2 hardware fixed.

Confirmed before full-serving comparison:
- Official base image digest: sha256:18372a7224938643461b846fb64c5c9d3d6e9727e82caf2dc3043e620c9d4d7a; local ARM64 image ID 785fd756b4ae1cacda7612cb859dbf227330e8ced9a6fdea42e9379128bebce0.
- The pristine image reproduces all 207 synthetic BF16 GEMM changes, max 0.03125, with serial repeat exact. No source overrides, network, or serving service were used.
- Real-checkpoint first HC mixer replay in that image matches captured patched-server serial and mixed-batch outputs bit-for-bit. No source overrides were used. The replay script only loads weights, captured input token IDs and executes upstream ops and torch linear; captured outputs are comparison targets, not computational inputs.
- Installed HC module, HC glue kernels and standard linear module hash-identically match upstream v0.29.0 commit 98dff2a81d747d1dba01a47f939f48c3526d4206.

Experiments:
1. `baseline`: existing complete serving configuration, fresh serial/concurrent/serial control.
2. `compute-off`: all optional BF16/MoE/PLE accelerations, custom RoCE all-reduce and FLA runtime dispatch override disabled. Paging and required PLE compatibility remain. This is not labeled pristine upstream.
3. `upstream` / `upstream-seed` / `cold-upstream-seed`: official base image with one explicit PLE loader adapter, without the fork's scheduler, paging, transport, compute, runner, or native QSA changes. Adapter selects upstream's existing global-scale FP8 PLE method for the checkpoint's declared mixed-ModelOpt FP8 table. Preflight confirms unmodified upstream selects no FP8 method for that declaration; the adapter selects the correct existing method. No arithmetic kernels are added or modified.

The loader adapter and all test commands are archived separately. Completed whole-response outcomes, attribution limits and final configuration are recorded in REPORT.md.
