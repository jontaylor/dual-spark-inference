"""Print a sampled live service summary without reading or displaying credentials."""
import argparse
import concurrent.futures
import datetime
import json
import subprocess
import time
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--interval', type=float, default=10)
args = parser.parse_args()
if args.interval <= 0:
    parser.error('interval must be positive')
cfg = json.loads((Path(__file__).resolve().parents[1] / 'deploy_config.json').read_text())

def metrics():
    with urllib.request.urlopen(f"http://127.0.0.1:{cfg['port']}/metrics", timeout=5) as response:
        lines = response.read().decode().splitlines()
    result = {}
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        name, value = line.rsplit(' ', 1)
        name = name.split('{')[0]
        result[name] = result.get(name, 0) + float(value)
    return result

def gpu(rank):
    command = ['nvidia-smi', '--query-gpu=utilization.gpu,power.draw', '--format=csv,noheader,nounits']
    if rank:
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', cfg['worker_ip'], *command]
    try:
        return subprocess.check_output(command, text=True, timeout=8).strip()
    except (subprocess.SubprocessError, OSError):
        return 'unavailable'

before = metrics()
start = time.monotonic()
time.sleep(args.interval)
after = metrics()
elapsed = time.monotonic() - start

def delta(name):
    name = 'vllm:' + name
    return max(0, after.get(name, 0) - before.get(name, 0))

drafted = delta('spec_decode_num_draft_tokens_total')
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    gpus = list(pool.map(gpu, [0, 1]))
print(json.dumps({
    'time': datetime.datetime.now().astimezone().isoformat(),
    'interval_seconds': round(elapsed, 2),
    'running': after.get('vllm:num_requests_running'),
    'waiting': after.get('vllm:num_requests_waiting'),
    'generated_tok_s': round(delta('generation_tokens_total') / elapsed, 2),
    'kv_usage_fraction': after.get('vllm:kv_cache_usage_perc'),
    'draft_acceptance_fraction': round(delta('spec_decode_num_accepted_tokens_total') / drafted, 4) if drafted else None,
    'preemptions_delta': delta('num_preemptions_total'),
    'spark1_gpu_util_percent_power_w': gpus[0],
    'spark2_gpu_util_percent_power_w': gpus[1],
}, indent=2))
