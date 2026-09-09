#!/usr/bin/env python3
"""Verify pinned HF files against published LFS hashes or Git blob hashes."""
import hashlib
import json
from runtime_config import load_config, model_snapshot, resolve_path
import sys
import time
import urllib.request
from pathlib import Path

root=Path(__file__).resolve().parent
cfg=load_config(root)
repo=cfg['model_id'];revision=cfg['revision']
data=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'))
assert data['sha']==revision
snapshot=(resolve_path(cfg['paths']['hf_cache'], root) / ('models--' + cfg['model_id'].replace('/', '--')) / 'snapshots')/revision
files=[]
deadline=time.monotonic()+7200
for entry in data['siblings']:
    path=snapshot/entry['rfilename'];size=entry['size']
    if '--wait-for-files' in sys.argv:
        if not path.exists(): print('Waiting for',entry['rfilename'],flush=True)
        while not path.exists() or path.stat().st_size!=size:
            if time.monotonic()>deadline:
                raise TimeoutError(f'Checkpoint download/transfer incomplete: {path}')
            time.sleep(2)
    assert path.stat().st_size==size,(path,size)
    if entry.get('lfs'):
        h=hashlib.sha256();expected=entry['lfs']['sha256'];algorithm='sha256'
    else:
        h=hashlib.sha1();h.update(f'blob {size}\0'.encode());expected=entry['blobId'];algorithm='git-blob-sha1'
    with path.open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):h.update(block)
    assert h.hexdigest()==expected,f'Checksum mismatch: {path}'
    files.append({'name':entry['rfilename'],'size':size,'hash':expected,'algorithm':algorithm})
    print('Verified',entry['rfilename'],size,flush=True)
(root/'verified-checkpoint.json').write_text(json.dumps({'model_id':repo,'revision':revision,'files':files},indent=2)+'\n')
print('All pinned checkpoint files verified',flush=True)
