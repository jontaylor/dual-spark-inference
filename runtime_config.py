"""Host-local paths and public configuration; credentials remain in a file."""
import json
import os
from pathlib import Path


def resolve_path(value, root):
    path = Path(value).expanduser()
    return path if path.is_absolute() else (Path(root) / path).resolve()


def load_config(root=None):
    root = Path(root or Path(__file__).resolve().parent)
    config_path = resolve_path(os.environ.get('DUAL_SPARK_CONFIG', 'deploy_config.json'), root)
    cfg = json.loads(config_path.read_text())
    if cfg['model_id'] != 'Qwen/Qwen3.8-Flash-Next-FP8':
        raise ValueError('This tested deployment requires the official Qwen FP8 checkpoint')
    if cfg['kv_cache_dtype'] != 'bfloat16':
        raise ValueError('This tested deployment requires BF16 KV')
    required = ('hf_cache', 'ple_cache', 'runtime_cache')
    for key in required:
        if not cfg.get('paths', {}).get(key):
            raise ValueError(f'Missing paths.{key}')
    return cfg


def model_snapshot(cfg, root):
    return (resolve_path(cfg['paths']['hf_cache'], root)
            / ('models--' + cfg['model_id'].replace('/', '--'))
            / 'snapshots' / cfg['revision'])
