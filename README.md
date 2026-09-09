# Dual-Spark inference

Reproducible deployment tooling for two DGX Spark / GB10 machines, derived from
[MiaAI Lab's dual-Spark setup](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks).
The inference changes live in the pinned
[vllm-gb10](https://github.com/jontaylor/vllm-gb10) `server` submodule.

## Selected configuration

| Setting | Value |
|---|---|
| Model | Official `Qwen/Qwen3.8-Flash-Next-FP8` |
| Model revision | `236dfdf285828023ca3bcd3f37366c58a3469b13` |
| KV cache | BF16 |
| Parallelism | TP2 + EP across two nodes |
| Speculation | MTP3, 98,304 draft IDs; full target verification |
| Maximum sequences | 4 |
| Maximum total tokens per request | 262,144, including output |
| Total scheduled-token budget | 8,192 |
| GPU memory utilisation | 0.76 |
| PLE | Each node's local NVMe, Mia's packed mmap format with FP8 rows |
| Graphs | Full decode graphs; profiling disabled |
| API port | 30001; credential supplied through a local file |

The 8,192-token budget is shared across eligible requests. Actual non-final
prefill chunks respect the 1,600-token state alignment; C1 typically receives
8,000 and four eligible prefills typically receive 1,600 each.
Optimisation flags enable code from the server submodule; stock vLLM with the
same settings does not reproduce this deployment.

After the recurrent-cache fix, cold and warm full-context C4 completed without
preemption. See [the capacity and throughput report](docs/cache-fix-2026-09-09.md).
The original end-to-end repair workload has not been rerun after these changes.

## Repository responsibilities

- This repository: configuration, services, checkpoint/table preparation,
  selected draft IDs, benchmark tools and aggregate reports.
- `server`: maintained inference source, native helpers, tests and Dockerfile.
- Host-local ignored files: `deploy_config.json`, verification records,
  generated `files/`, credentials, checkpoints, packed tables and raw captures.

The server repository records an unresolved upstream Git revision in the pinned
release image. Its exact affected modules are preserved and hash-verified;
the first release is an image-based source-overlay build, not a complete
from-source reconstruction of the original vLLM image.

## Prepare each node

Requirements: Linux/aarch64 on GB10, working NVIDIA Docker GPU support, CUDA
libraries from the pinned image, a C compiler with OpenMP, Python 3.12, and
working RDMA over the selected interconnect. Passwordless SSH from head to
worker and permission to start/stop the worker unit are required for supervision.
Allow roughly 8 GiB/node for the OS. Model weights and a local packed PLE table
must fit on each node's NVMe; they are not downloaded by cloning this repository.

```bash
git clone --recurse-submodules https://github.com/jontaylor/dual-spark-inference.git
cd dual-spark-inference
cp deploy_config.example.json deploy_config.json
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Edit `deploy_config.json` on both nodes: head/worker addresses, interface/HCA,
cache paths and the head's existing `api_key_file`. The addresses in the example
are placeholders. Keep the same model revision and server submodule commit on
both nodes. For an existing endpoint, retain its port, served names and key
file. Credential contents never belong in configuration or Git.

Run on each node:

```bash
.venv/bin/python download_model.py
.venv/bin/python verify_checkpoint.py
.venv/bin/python build_ple_table.py
.venv/bin/python verify_packed_rows.py
.venv/bin/python prepare_local.py
```

Alternatively, transfer the complete model cache and packed table from head to
worker, preserving Hugging Face blob/snapshot links, then run the verification
steps locally on the worker. Both nodes must have their own NVMe-backed table.
Checkpoint verification reads all model files and can take several minutes.
Do not regenerate files that are mounted into a running service: use a separate
checkout for preparation and switch releases during a controlled restart.

`prepare_local.py --sources-only` exports source without loading weights,
building helpers or starting a service. Normal preparation builds helpers
locally using the pinned image and prepares checkpoint metadata overlays.
The cached checkpoint itself is not modified.

## Review and start services

Render units on each node, then review the head unit on the head and worker
unit on the worker. The checkout path may differ between nodes.

```bash
python3 render_services.py
python3 launch_rank.py 0 --dry-run  # use rank 1 on the worker
```

On the worker:

```bash
sudo install -m 644 rendered-services/qwen38-next-qwen-fp8-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
```

On the head, after stopping the outgoing inference service to free its memory:

```bash
sudo install -m 644 rendered-services/qwen38-next-qwen-fp8.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start qwen38-next-qwen-fp8.service
```

The head starts the worker automatically. The supervisor stops the pair after
sustained host-memory pressure rather than repeatedly restarting under load.
Model startup normally takes around 11–12 minutes on the measured setup.
After readiness, run `.venv/bin/python smoke_api.py results/smoke.json`
(create `results/` first). Services are not automatically enabled at boot.

## Optional image with the changes included

The default uses the already-validated source-overlay layout. A Dockerfile is
also provided so the modifications and native helpers can be included in an
image rather than supplied as individual runtime mounts:

```bash
docker build --build-arg SOURCE_REVISION="$(git -C server rev-parse HEAD)" \
  -t vllm-gb10:local server
bash server/tools/test_image.sh vllm-gb10:local
```

Transfer that image to the worker and verify matching image IDs. Set `image`
to its local tag or immutable registry digest and `server_in_image` to `true`
on both nodes. Checkpoint metadata, credentials, draft IDs and the PLE table
remain host inputs. This image has build and CPU-regression coverage; it has
not replaced the measured production deployment or undergone a new full-model
serving test. Changing image/weights/shapes requires revalidation.

## Benchmark and retain a release

Supply a local source corpus that you are authorised to use. The runner records
outputs locally under ignored `results/`; inspect them before sharing.

```bash
.venv/bin/python bench_agent_context.py --corpus /path/to/source \
  --context 65000 --output 1024 --concurrency 1,2,4 --results ./results
```

Measure prompt processing separately from the common decode interval, and
check prefix-cache hits and preemptions. The selected C4 setting intentionally
caps concurrency; higher C8–C24 experiments need separate capacity validation.

Pin the deployment release tag, its server submodule commit, base/custom image
digest, model revision and local configuration together. Keep each release in
its own checkout. Rollback means stopping the service and pointing its units
back at the previously validated checkout/configuration, then restarting.

See [NOTICE.md](NOTICE.md) for Mia/vLLM attribution. This repository is
AGPL-3.0-or-later; model checkpoints and the container retain their own terms.
