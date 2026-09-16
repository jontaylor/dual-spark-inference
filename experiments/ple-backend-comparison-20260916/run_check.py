"""Run only with serving stopped: a separate CUDA context is required."""

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
PKG = "/usr/local/lib/python3.12/dist-packages/vllm/"
cfg = json.loads((BASE / "deploy.before-r0.json").read_text())
state = subprocess.run(["docker", "inspect", "--format", "{{.State.Running}}",
                        "qwen38-kv-paging-r0"], capture_output=True, text=True)
if state.stdout.strip() == "true":
    raise SystemExit("Stop serving before the isolated GPU integration check")
cmd = ["docker", "run", "--rm", "--name", "ple-backend-check", "--gpus", "all",
       "--network", "none", "--ipc", "host", "--ulimit", "memlock=-1",
       "--security-opt", "seccomp=" + cfg["ple_io_uring_seccomp"]]
env = {
    "GB10_PLE_IN_PROCESS": "1", "GB10_PLE_MAPPED_TRANSPORT": "1",
    "VLLM_PLE_CPU_OFFLOAD": "1", "VLLM_PLE_LOCAL_TP": "1",
    "GB10_PLE_ROW_CACHE_MB": "1", "GB10_PLE_GATHER_THREADS": "16",
    "GB10_PLE_NATIVE_HASH": "1", "OMP_WAIT_POLICY": "PASSIVE",
    "VLLM_PLE_PACKED_TABLE_DIR": "/tmp/ple-test-table",
    "GB10_PLE_GPU_HASH": "1", "GB10_PLE_OVERLAP": "1",
    "GB10_PLE_SUBMISSION_POLICY": "sqpoll",
    "GB10_PLE_BACKEND_CONTROL": "/tmp/backend-control.bin",
    "GB10_PLE_READ_CONTROL": "/tmp/read-policy.bin",
    "GB10_PLE_COMPARISON_LOG_DIR": "/tmp/ple-logs",
}
for key, value in env.items():
    cmd += ["-e", f"{key}={value}"]
mounts = {PKG + key: value for key, value in cfg["runtime_overrides"].items()}
for name in ("in_process", "backend_comparison", "backend_control"):
    mounts[PKG + f"v1/ple_offload/{name}.py"] = BASE / f"{name}.py"
for name in ("connector", "worker", "protocol"):
    mounts[PKG + f"v1/ple_offload/{name}.py"] = BASE / f"legacy_{name}.py"
for key, filename in (("ple_batch_library", "libple_batch_reader.so"),
                      ("ple_hash_library", "libple_hash_gpu.so"),
                      ("ple_mapped_wait_library", "libmapped_wait.so")):
    mounts["/opt/gb10/" + filename] = cfg[key]
mounts["/check.py"] = BASE / "check_backends.py"
for destination, source in mounts.items():
    cmd += ["-v", f"{source}:{destination}:ro"]
cmd += ["--entrypoint", "/bin/sh", cfg["image"], "-c",
        "uv venv --offline --system-site-packages /tmp/check-venv && "
        "/tmp/check-venv/bin/python /check.py"]
log = ROOT / "results/ple-backend-comparison-20260916/integration.log"
with log.open("w") as handle:
    result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT)
print(log)
raise SystemExit(result.returncode)
