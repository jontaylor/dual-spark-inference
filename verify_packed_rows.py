#!/usr/bin/env python3
"""Compare rows from every packed FP8 shard with the pinned original tensor."""
import json
from runtime_config import load_config, model_snapshot, resolve_path
import os
import random
import re
import struct
from pathlib import Path

root=Path(__file__).resolve().parent
cfg=load_config(root)
snap=(resolve_path(cfg['paths']['hf_cache'], root) / ('models--' + cfg['model_id'].replace('/', '--')) / 'snapshots')/cfg['revision']
packed_dir=resolve_path(cfg['paths']['ple_cache'], root)/cfg['revision']
index=json.loads((snap/'model.safetensors.index.json').read_text())['weight_map']
table=next(packed_dir.glob('*.packed_u8'))
meta=json.loads(Path(str(table)+'.json').read_text())
assert meta['row_format']=='fp8_e4m3' and meta['snapshot']==cfg['revision']
headers={};rng=random.Random(42);checked=0
with table.open('rb') as packed:
    for name,filename in index.items():
        m=re.search(r'\.ngram_embedding\.shard_(\d+)\.weight$',name)
        if not m:continue
        shard=int(m[1]);path=snap/filename
        if filename not in headers:
            with path.open('rb') as f:
                n=struct.unpack('<Q',f.read(8))[0];headers[filename]=(json.loads(f.read(n)),8+n)
        h,base=headers[filename];info=h[name]
        rows,width=info['shape']
        assert info['dtype']=='F8_E4M3' and width==meta['row_width'] and rows==meta['rows_per_shard']
        with path.open('rb') as original:
            for row in [0,rows-1,*[rng.randrange(rows) for _ in range(14)]]:
                expected=os.pread(original.fileno(),width,base+info['data_offsets'][0]+row*width)
                actual=os.pread(packed.fileno(),width,(shard*rows+row)*width)
                assert actual==expected and len(actual)==width,(shard,row)
                checked+=1
assert checked==meta['num_shards']*16
result={'revision':cfg['revision'],'row_format':meta['row_format'],'rows_checked':checked,
        'shards_checked':meta['num_shards'],'packed_bytes':table.stat().st_size,'status':'pass'}
(root/'verified-packed-rows.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
