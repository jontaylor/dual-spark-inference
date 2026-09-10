import hashlib
import json
from types import SimpleNamespace as NS

import pytest

import paging_ops as ops


def config(tmp_path):
    return {'image': 'tested', 'image_ids': ['image0', 'image1'],
            'kv_paging': {'disk_root': str(tmp_path / 'vllm-gb10-kv-paging'),
                          'disk_bytes_per_rank': 100}}


def test_live_container_prevents_cache_cleanup(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    monkeypatch.setattr(ops.subprocess, 'run', lambda *a, **k: NS(returncode=0, stdout='true'))
    with pytest.raises(RuntimeError, match='Stop the running container'):
        ops.prepare_disk(cfg, tmp_path, 0)
    assert not (tmp_path / 'vllm-gb10-kv-paging').exists()


def test_unmarked_or_unknown_directory_cannot_be_deleted(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cache = tmp_path / 'vllm-gb10-kv-paging'
    cache.mkdir()
    (cache / 'unrelated').mkdir()
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return NS(returncode=1, stdout='')
    monkeypatch.setattr(ops.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='ownership marker'):
        ops.prepare_disk(cfg, tmp_path, 0)
    (cache / '.same-process-paging').write_text('owned')
    with pytest.raises(RuntimeError, match='Unexpected file'):
        ops.prepare_disk(cfg, tmp_path, 0)
    assert all(command[0] == 'docker' for command in calls)


def test_expired_uuid_cleanup_is_scoped_and_disk_headroom_enforced(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    cache = tmp_path / 'vllm-gb10-kv-paging'
    cache.mkdir()
    (cache / '.same-process-paging').write_text('owned')
    expired = cache / 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
    expired.mkdir()
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return NS(returncode=1, stdout='')
    monkeypatch.setattr(ops.subprocess, 'run', run)
    monkeypatch.setattr(ops.shutil, 'disk_usage', lambda p: NS(free=100))
    with pytest.raises(RuntimeError, match='Insufficient free disk'):
        ops.prepare_disk(cfg, tmp_path, 0)
    assert calls[-1] == ['sudo', '-n', '/usr/bin/rm', '-rf', '--', str(expired)]


def test_modified_bundle_is_rejected_before_launch(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    (tmp_path / 'overlay.py').write_bytes(b'original')
    (tmp_path / 'paging-manifest.json').write_text(json.dumps({
        'sha256': {'overlay.py': hashlib.sha256(b'original').hexdigest()}}))
    monkeypatch.setattr(ops.subprocess, 'check_output', lambda *a, **k: 'image0\n')
    ops.verify_runtime(cfg, tmp_path, 0)
    with pytest.raises(RuntimeError, match='image differs'):
        ops.verify_runtime(cfg, tmp_path, 1)
    (tmp_path / 'overlay.py').write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='overlay differs'):
        ops.verify_runtime(cfg, tmp_path, 0)
