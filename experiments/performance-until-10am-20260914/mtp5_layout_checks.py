from __future__ import annotations
import ast,json,math,torch
from types import SimpleNamespace as N
from pathlib import Path
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator,MambaStateShapeCalculator
from vllm.v1.kv_cache_interface import MambaSpec,FullAttentionSpec,CircularBufferSpec
root=Path('/experiments/performance-until-10am-20260914');tree=ast.parse((root.parent/'targeted-determinism-20260913/model.final.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen4ExpForCausalLM')
names={'get_gdn_mamba_state_shape_from_config','get_gdn_mamba_state_dtype_from_config','get_ple_mamba_state_shape_from_config','get_ple_mamba_state_dtype_from_config','get_mamba_specs_from_config'}
cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names];cls.bases=[];cls.decorator_list=[]
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[]);exec(compile(ast.fix_missing_locations(module),'actual_model_shapes','exec'))
hf=N(**json.loads(Path('/model-config.json').read_text())['text_config']);rows=[]
for depth in (3,5):
 cfg=N(model_config=N(dtype=torch.bfloat16,hf_text_config=hf),cache_config=N(mamba_cache_dtype='auto',mamba_ssm_cache_dtype='float32'),parallel_config=N(tensor_parallel_size=2),speculative_config=N(num_speculative_tokens=depth))
 specs=Qwen4ExpForCausalLM.get_mamba_specs_from_config(cfg);size=max(s.page_size_bytes for s in specs);attn=FullAttentionSpec(block_size=1,num_kv_heads=1,head_size=hf.head_dim,dtype=torch.bfloat16).page_size_bytes
 for base in (1600,1920):
  effective=base*math.ceil(size/(base*attn));capacity=hf.indexer_compress_ratio*math.ceil((hf.indexer_compress_ratio+depth)/hf.indexer_compress_ratio)
  rows.append({'depth':depth,'requested_block':base,'effective_block':effective,'mamba_bytes':size,'attention_bytes_per_token':attn,'ring_capacity':capacity,'divisible':effective%capacity==0,'grid64':effective%64==0,'padding_fraction':effective*attn/size-1})
assert next(r for r in rows if r['depth']==5 and r['requested_block']==1600)['divisible'] is False
assert next(r for r in rows if r['depth']==5 and r['requested_block']==1920)['divisible'] is True
print(json.dumps({'rows':rows,'scope':'Actual model shape calculators and cache specs, no model weights or GPU. Platform rounding formula reproduced; full startup still required.'},indent=2))
