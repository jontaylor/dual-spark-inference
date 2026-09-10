"""Freeze the committed vLLM paging overlay into a verifiable deployment bundle."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--server-root', required=True, type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parent
source = args.server_root.resolve()
files = {
    'files/paging/parking_scheduler_base.py': 'vllm/v1/core/sched/scheduler.py',
    'files/paging/parking_policy.py': 'vllm/v1/core/sched/parking_policy.py',
    'files/paging/gb10_parking_scheduler.py': 'vllm/v1/core/sched/gb10_parking_scheduler.py',
    'files/paging/aligned_connector.py': 'vllm/distributed/kv_transfer/kv_connector/v1/gb10_aligned_offloading_connector.py',
    'files/paging/rank_local_disk.py': 'vllm/v1/kv_offload/rank_local_disk.py',
    'files/modelopt_patched.py': 'vllm/model_executor/layers/quantization/modelopt.py',
}
revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
hashes = {}
for destination, relative in files.items():
    path = root / destination
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = (source / relative).read_bytes()
    committed = subprocess.check_output(['git', '-C', str(source), 'show', f'{revision}:{relative}'])
    if contents != committed:
        raise RuntimeError(f'Commit the source change before bundling: {relative}')
    shutil.copy2(source / relative, path)
    hashes[destination] = hashlib.sha256(contents).hexdigest()
for name in ('config_patched.json', 'hf_quant_config_patched.json'):
    path = root / 'files' / name
    hashes[f'files/{name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / 'paging-manifest.json').write_text(json.dumps(
    {'server_revision': revision, 'sha256': hashes}, indent=2) + '\n')
print(f'Prepared {len(hashes)} overlays from {revision}')
