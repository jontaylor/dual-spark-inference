"""Prepare and dry-run a profiler-enabled baseline; never restart or deploy."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
rank = int(sys.argv[1])
assert rank in (0, 1)
base = json.loads((root / 'deploy_config.json').read_text())
cfg = copy.deepcopy(base)
cfg['profiling'].update(enabled=True, max_iterations=40,
                        output_root=str(root / 'results/async-scheduling-20260916/kernel-profiles'))
assert {k:v for k,v in cfg.items() if k != 'profiling'} == {k:v for k,v in base.items() if k != 'profiling'}
out = root / 'results/async-scheduling-20260916'
out.mkdir(parents=True, exist_ok=True)
path = out / f'proposed-profile-rank{rank}.json'
path.write_text(json.dumps(cfg, indent=2)+'\n')
env = dict(os.environ, DUAL_SPARK_CONFIG=str(path))
run = subprocess.run([sys.executable, str(root/'launch_rank.py'), str(rank), '--dry-run'],
                     env=env, capture_output=True, text=True, check=True)
rows = [json.loads(line) for line in run.stdout.splitlines()]
args = rows[0]['args']
profile = json.loads(args[args.index('--profiler-config')+1])
assert profile == {'profiler':'cuda', 'max_iterations':40}
assert '--no-async-scheduling' in args
assert args[args.index('--scheduler-cls')+1].endswith('.GB10ParkingScheduler')
summary = dict(rank=rank, dry_run_passed=True, deployed=False,
               proposed_config=str(path), only_changed_section='profiling',
               profiler=profile, synchronous_baseline=True)
(out / f'profile-preflight-rank{rank}.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary))
