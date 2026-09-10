"""Validation and lifecycle cleanup for the dedicated process-local disk cache."""
import hashlib
import json
import shutil
import subprocess
import uuid
from pathlib import Path

from runtime_config import resolve_path


def verify_runtime(cfg, root, rank):
    manifest = json.loads((root / 'paging-manifest.json').read_text())
    for relative, expected in manifest['sha256'].items():
        path = root / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Runtime overlay differs from manifest: {relative}')
    actual = subprocess.check_output(
        ['docker', 'image', 'inspect', cfg['image'], '--format', '{{.Id}}'], text=True
    ).strip()
    if actual != cfg['image_ids'][rank]:
        raise RuntimeError(f'Rank {rank} image differs from the tested image')


def prepare_disk(cfg, root, rank):
    cache = resolve_path(cfg['kv_paging']['disk_root'], root)
    if cache.name != 'vllm-gb10-kv-paging' or cache.is_symlink():
        raise RuntimeError('Refusing cleanup outside the dedicated paging directory')
    for name in (f'qwen38-kv-paging-r{rank}', f'nvfp4-paging-test-r{rank}',
                 f'qwen38-next-qwen-fp8-r{rank}'):
        status = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', name],
                                capture_output=True, text=True)
        if status.returncode == 0 and status.stdout.strip() == 'true':
            raise RuntimeError(f'Stop the running container first: {name}')
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    marker = cache / '.same-process-paging'
    entries = list(cache.iterdir())
    if entries and not marker.is_file():
        raise RuntimeError('Existing directory has no paging ownership marker')
    if not marker.exists():
        marker.write_text('Metadata is engine-local; UUID directories expire on restart.\n')
    for path in entries:
        if path == marker:
            continue
        try:
            uuid.UUID(path.name)
        except ValueError as error:
            raise RuntimeError(f'Unexpected file in cache root: {path.name}') from error
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError(f'Unexpected cache entry: {path.name}')
        # Docker writes these directories as root. Only expired engine UUIDs
        # under this dedicated, marked directory can reach this command.
        subprocess.run(['sudo', '-n', '/usr/bin/rm', '-rf', '--', str(path)], check=True)
    needed = cfg['kv_paging']['disk_bytes_per_rank'] + 8 * 2**30
    if shutil.disk_usage(cache).free < needed:
        raise RuntimeError('Insufficient free disk for the configured cache plus 8 GiB')
