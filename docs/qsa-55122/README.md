# Deployed QSA top-k backport

The September 12 deployment uses the deterministic kernel from upstream
[vLLM PR #55122](https://github.com/vllm-project/vllm/pull/55122).
See [RESULTS.md](RESULTS.md) for correctness results, performance limitations,
and rollback details. This directory is a historical deployment record;
image IDs are node-specific, and readiness observations are from that rollout.

## Source and build

The exact native source is in
[jontaylor/vllm-gb10 at 20496bfca8](https://github.com/jontaylor/vllm-gb10/commit/20496bfca8),
on branch `gb10/qsa-deterministic-55122`. The active paging fork also contains
the backport at `01b52a02e5` on `gb10-kv-paging`.
Build the native artifact from the former commit: the latter contains additional
fork changes that were not part of this native build.

The deployed binary hash and toolchain versions are in
[native-provenance.json](native-provenance.json). Binaries, compiler objects,
CUDA dependencies and virtual environments are deliberately excluded from Git.

For a rebuild, coordinate the workload and stop both inference services first
to reclaim unified RAM. Use a CUDA 13 development environment based on the
original node image, with GCC 13.3, CMake 3.31.6, Ninja and git available.
Create `/tmp/qsa-venv` using `uv venv --system-site-packages` so it can use the
base image's PyTorch 2.13.0+cu130. Install build requirements with `uv pip`
into that environment. Mount the pinned source at `/src` and a writable copy
of this directory at `/artifacts`; provide CUTLASS v4.4.2 sources at
`/artifacts/deps/cutlass`. Run:

```bash
QSA_BUILD_JOBS=4 bash /artifacts/build-native.sh
```

Choose parallelism for available RAM. Only one job was validated for the
original 12 GiB builder limit; two jobs exceeded that limit. Increase the
builder memory allocation after shutting down inference. This script builds
only `_C_stable_libtorch`, retaining the base image's Python and MoE library.
It is GB10-specific, not a general multiarchitecture wheel.

From the artifact directory, package separately on each node using its own
stock image ID from `image-identities.json`:

```bash
docker build -f Dockerfile.runtime --build-arg BASE_IMAGE=<node-stock-image-id> \
  -t vllm-gb10:v029-roce-qsa55122 .
```

A rebuild can produce different binary/image hashes. Regenerate provenance and
image identities, validate both nodes, then use `set_image_arm.py candidate
--config <live-config> --identities <identities>` to update only image fields.
Restart both ranks together and verify readiness. Never replace a loaded library.
The original artifacts remain at `/home/jon/qsa-backport-assessment-20260912`.

## Validation record

The upstream test file is in the pinned vLLM checkout at
`tests/kernels/test_top_k_per_row.py`; additional graph/paged-QSA checks and
kernel measurements are in `validation/`. Run GPU tests in the candidate
runtime using a uv-managed environment, as in the original execution.
The serving probe defaults to the local deployment path and is intended for
this two-node installation. Preserve representative workload settings and
coordinate timing probes before comparing results.

All source hooks passed except two explicitly exempted checks on pinned
upstream code: `typos` flags CUDA's `optin` spelling, and
`check-torch-cuda-call` flags the upstream test's CUDA API call. Keeping these
unchanged preserves the source hashes of the artifact tested on both GPUs.
