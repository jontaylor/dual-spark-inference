#!/usr/bin/env python3
"""Prepare this checkout without starting or stopping any service."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from runtime_config import load_config, model_snapshot, resolve_path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sources-only', action='store_true', help='Export code without building libraries or reading weights')
args = parser.parse_args()
cfg = load_config(root)
server = root / 'server'
if not (server / 'overlays.json').is_file():
    raise SystemExit('Run git submodule update --init --recursive first')
base = json.loads((server / 'provenance/base-image.json').read_text())['base_image']
if not cfg.get('server_in_image') and cfg['image'] != base:
    raise SystemExit('Source overlays require the exact pinned base image')
subprocess.run([sys.executable, str(server / 'tools/export_overlays.py'),
                '--output', str(root / 'files')], check=True)
for p in (root / 'model_tools').glob('*.py'):
    shutil.copy2(p, root / 'files' / p.name)
if args.sources_only:
    raise SystemExit(0)
if not cfg.get('server_in_image'):
    subprocess.run(['bash', str(root / 'files/gb10/build.sh')], check=True)
    if cfg.get('optimizations', {}).get('ple_mapped_transport'):
        subprocess.run(['docker', 'run', '--rm', '--network', 'none',
                        '-v', str(root / 'files/gb10') + ':/work', '--entrypoint', 'python3',
                        base, '/work/build_mapped.py'], check=True)
snapshot = model_snapshot(cfg, root)
if not (snapshot / 'config.json').is_file():
    raise SystemExit('Source preparation completed; download the pinned checkpoint before preparing model configuration')
subprocess.run([sys.executable, str(root / 'model_tools/patch_checkpoint_config.py'),
                str(snapshot), str(root / 'files')], check=True)
dtype = subprocess.check_output([sys.executable, str(root / 'model_tools/detect_ple_dtype.py'),
                                str(snapshot)], text=True).strip()
if dtype != 'float8_e4m3fn':
    raise SystemExit(f'Unexpected PLE dtype: {dtype}')
print('Prepared source and model configuration. Service startup is separate.')
