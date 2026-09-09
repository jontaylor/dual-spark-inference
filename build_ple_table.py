#!/usr/bin/env python3
"""Build Mia's packed mmap table using this deployment's configured paths."""
import subprocess
import sys
from pathlib import Path
from runtime_config import load_config, model_snapshot, resolve_path

root = Path(__file__).resolve().parent
cfg = load_config(root)
out = resolve_path(cfg['paths']['ple_cache'], root) / cfg['revision']
subprocess.run([sys.executable, str(root / 'model_tools/build_ple_packed_table.py'),
                str(model_snapshot(cfg, root)), str(out)], check=True)
