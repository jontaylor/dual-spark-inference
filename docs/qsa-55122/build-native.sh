#!/usr/bin/env bash
set -euo pipefail
# Run in the resource-limited builder with source at /src and artifacts at /artifacts.
export PATH=/tmp/qsa-venv/bin:$PATH
export TORCH_CUDA_ARCH_LIST=12.1
cd /tmp
cmake -S /src -B /artifacts/build -G Ninja \
  -DVLLM_PYTHON_EXECUTABLE=/tmp/qsa-venv/bin/python \
  -DCMAKE_BUILD_TYPE=Release -DNVCC_THREADS=1 \
  -DVLLM_CUTLASS_SRC_DIR=/artifacts/deps/cutlass \
  -DVLLM_QSA_BACKPORT_NATIVE_ONLY=ON \
  -DCUDA_NVRTC_LIB=/usr/local/cuda/targets/sbsa-linux/lib/libnvrtc.so.13 \
  -DCMAKE_INSTALL_PREFIX=/artifacts/staged
cmake --build /artifacts/build --target _C_stable_libtorch -j "${QSA_BUILD_JOBS:-1}"
cmake --install /artifacts/build --component _C_stable_libtorch
