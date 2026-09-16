"""Change both hosts' backend selection while the repeating load is drained."""

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time
import urllib.request

from backend_control import decode, encode

ROOT = Path(__file__).resolve().parents[2]
CONTROL = "/home/jon/.cache/vllm-ple-control/ple-backend-policy.bin"
PYTHON = str(ROOT / ".venv/bin/python")
REMOTE = "jon@192.168.100.11"
ACCESS = '''
import ctypes,json,mmap,os,sys
fd=os.open(sys.argv[1],os.O_RDWR)
assert os.fstat(fd).st_size==4096
m=mmap.mmap(fd,4096,access=mmap.ACCESS_WRITE)
cell=ctypes.c_uint64.from_buffer(m)
lib=ctypes.CDLL('libatomic.so.1')
load=lib.__atomic_load_8;load.argtypes=[ctypes.c_void_p,ctypes.c_int];load.restype=ctypes.c_uint64
store=lib.__atomic_store_8;store.argtypes=[ctypes.c_void_p,ctypes.c_uint64,ctypes.c_int]
before=load(ctypes.addressof(cell),2)
if len(sys.argv)>2:
 store(ctypes.addressof(cell),int(sys.argv[2]),3)
 m.flush()
after=load(ctypes.addressof(cell),2)
print(json.dumps({'before':before,'after':after}))
del cell
m.close();os.close(fd)
'''


def access(rank, value=None):
    argv = [PYTHON, "-c", ACCESS, CONTROL]
    if value is not None:
        argv.append(str(value))
    if rank:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                REMOTE, shlex.join(argv)]
    return json.loads(subprocess.check_output(argv, text=True, timeout=20))


def idle():
    config = json.loads((ROOT / "deploy_config.json").read_text())
    key = Path(config["api_key_file"]).read_text().strip()
    request = urllib.request.Request(
        f"http://127.0.0.1:{config['port']}/metrics",
        headers={"Authorization": "Bearer " + key},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        metrics = response.read().decode()
    values = {}
    for kind in ("running", "waiting"):
        matches = re.findall(
            rf"^vllm:num_requests_{kind}(?:\{{[^\n]*\}})?\s+([^\s]+)",
            metrics, re.MULTILINE,
        )
        if not matches:
            raise RuntimeError(f"Missing {kind} request gauge")
        values[kind] = sum(float(value) for value in matches)
    if any(value != 0 for value in values.values()):
        raise RuntimeError(f"Drain the repeating load before switching: {values}")
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("in_process", "external", "abba", "baab"))
    parser.add_argument("--block-steps", type=int, default=128)
    parser.add_argument("--label", default="manual")
    args = parser.parse_args()
    before = [access(rank)["after"] for rank in (0, 1)]
    if before[0] != before[1]:
        raise RuntimeError("Host control words disagree; investigate before testing")
    current = decode(before[0], 1)
    if args.mode is None:
        print(json.dumps({"control_word": before[0], **current}, indent=2))
        return
    idle()
    time.sleep(1)
    idle()
    value = encode(args.mode, current["epoch"] + 1, args.block_steps)
    start = time.time_ns()
    try:
        for rank in (1, 0):
            assert access(rank, value)["after"] == value
        assert all(access(rank)["after"] == value for rank in (0, 1))
    except BaseException:
        for rank in (0, 1):
            try:
                access(rank, before[rank])
            except Exception:
                pass
        raise
    record = {"event": "backend_change", "label": args.label,
              "start_ns": start, "end_ns": time.time_ns(),
              "before": before[0], "control_word": value,
              "requested_mode": args.mode, **decode(value, 1)}
    destination = ROOT / "results/ple-backend-comparison-20260916/control.jsonl"
    with destination.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
