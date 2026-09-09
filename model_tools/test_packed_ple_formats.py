"""Check MiaAI's builder and mmap attachment for NVFP4 and NVIDIA FP8 rows."""
import ast
import json
import logging
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

root=Path(__file__).resolve().parent
source=ast.parse((root/'ple_offload/worker.py').read_text())
cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='PleOffloadRunner')
attach=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_attach_packed_table')
attach.decorator_list=[]
namespace={'torch':torch,'json':json,'os':os,'PleOffloadLayer':object,'logger':logging.getLogger('test')}
exec(compile(ast.fix_missing_locations(ast.Module(body=[attach],type_ignores=[])),'attach','exec'),namespace)

def write_tensors(path,tensors):
    header={};raw=[];offset=0
    for name,dtype,array in tensors:
        payload=array.tobytes()
        header[name]={'dtype':dtype,'shape':list(array.shape),'data_offsets':[offset,offset+len(payload)]}
        offset+=len(payload);raw.append(payload)
    encoded=json.dumps(header).encode()
    path.write_bytes(struct.pack('<Q',len(encoded))+encoded+b''.join(raw))

with tempfile.TemporaryDirectory() as tmp:
    for fmt in ['nvfp4','fp8_e4m3']:
        snap=Path(tmp)/fmt;snap.mkdir();out=snap/'packed'
        prefix='model.language_model.layers.1.ple.ple_embedding.ngram_embedding'
        tensors=[];expected=[]
        for i in range(2):
            codes=np.arange(i*6,(i+1)*6,dtype=np.uint8).reshape(3,2)
            tensors.append((f'{prefix}.shard_{i}.weight','U8' if fmt=='nvfp4' else 'F8_E4M3',codes))
            if fmt=='nvfp4':
                scales=np.full((3,1),56+i,dtype=np.uint8)
                tensors.append((f'{prefix}.shard_{i}.weight_scale','F8_E4M3',scales))
                expected.append(np.concatenate([codes,scales],axis=1))
            else:expected.append(codes)
        if fmt=='fp8_e4m3':tensors.append((prefix+'.weight_scale','F32',np.array([0.25],dtype=np.float32)))
        write_tensors(snap/'model.safetensors',tensors)
        (snap/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{n:'model.safetensors' for n,_,_ in tensors}}))
        subprocess.run([sys.executable,str(root/'build_ple_packed_table.py'),str(snap),str(out)],check=True)
        table=next(out.glob('*.packed_u8'));meta=json.loads(Path(str(table)+'.json').read_text())
        expected=np.concatenate(expected)
        assert table.read_bytes()==expected.tobytes() and meta['row_format']==fmt
        weight_dtype=torch.float8_e4m3fn if fmt=='fp8_e4m3' else torch.uint8
        scale=torch.nn.Parameter(torch.tensor([0.25]) if fmt=='fp8_e4m3' else torch.empty((6,1),dtype=torch.float8_e4m3fn),requires_grad=False)
        emb=SimpleNamespace(weight=torch.nn.Parameter(torch.empty((6,2),dtype=weight_dtype),requires_grad=False),
            embedding_dim=2,weight_scale=scale,quant_method=SimpleNamespace(packed_row_width=3) if fmt=='nvfp4' else SimpleNamespace())
        namespace['_attach_packed_table']('test',SimpleNamespace(ngram_embedding=emb),str(table))
        assert emb.weight.numel()==0
        ids=torch.tensor([5,0,3,2,5])
        output=torch.empty((5,expected.shape[1]),dtype=weight_dtype)
        torch.index_select(emb._packed_table,0,ids,out=output.view(torch.uint8))
        assert np.array_equal(output.view(torch.uint8).numpy(),expected[ids.numpy()])
        if fmt=='fp8_e4m3':
            assert emb.weight_scale is scale and scale.item()==0.25
            reference=torch.from_numpy(expected[ids.numpy()]).view(torch.float8_e4m3fn)
            assert torch.equal(output.to(torch.bfloat16)*scale.to(torch.bfloat16),reference.to(torch.bfloat16)*scale.to(torch.bfloat16))
        else:assert emb.weight_scale.numel()==0
        print('PASS',fmt,'packed bytes, cross-shard lookup, mmap attachment, scale preservation',flush=True)
