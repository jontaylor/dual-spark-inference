#!/usr/bin/env python3
"""Download only the configured official checkpoint at its pinned revision."""
from pathlib import Path
from huggingface_hub import snapshot_download
from runtime_config import load_config, resolve_path

root = Path(__file__).resolve().parent
cfg = load_config(root)
snapshot_download(repo_id=cfg['model_id'], revision=cfg['revision'],
                  cache_dir=str(resolve_path(cfg['paths']['hf_cache'], root)))
