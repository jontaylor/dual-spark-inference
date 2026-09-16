from pathlib import Path
import json,contextlib
from unittest.mock import patch
import ast,json,math,torch
from types import SimpleNamespace as N
from pathlib import Path
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator,MambaStateShapeCalculator
from vllm.v1.kv_cache_interface import MambaSpec,FullAttentionSpec,CircularBufferSpec
root=Path('/experiments/performance-until-10am-20260914');tree=ast.parse(Path('/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen4ExpForCausalLM')
names={'get_gdn_mamba_state_shape_from_config','get_gdn_mamba_state_dtype_from_config','get_ple_mamba_state_shape_from_config','get_ple_mamba_state_dtype_from_config','get_mamba_specs_from_config'}
cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names];cls.bases=[];cls.decorator_list=[]
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),cls],type_ignores=[]);exec(compile(ast.fix_missing_locations(module),'actual_model_shapes','exec'))
hf=N(**json.loads(Path('/model-store/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47/config.json').read_text())['text_config']);rows=[]

from vllm.platforms.interface import Platform
from vllm.v1.attention.backend import MultipleOf
from vllm.model_executor.models import ModelRegistry
class Backend:
 @staticmethod
 def get_supported_kernel_block_sizes():return [MultipleOf(16)]
 @staticmethod
 def customize_spec(s):return s
out=[]
for depth in [3,5]:
 for block in [16,32,64,128,256,512,1600,1664,1920]:
  c=N(block_size=block,cache_dtype='auto',mamba_cache_mode='align',mamba_block_size=64,user_specified_mamba_block_size=True,mamba_page_size_padded=None,mamba_cache_dtype='auto',mamba_ssm_cache_dtype='float32')
  m=N(dtype=torch.bfloat16,hf_text_config=hf,use_mla=False,architecture='Qwen4ExpForCausalLM',get_num_kv_heads=lambda _:1,get_head_size=lambda:hf.head_dim)
  cfg=N(model_config=m,cache_config=c,parallel_config=N(tensor_parallel_size=2),speculative_config=N(num_speculative_tokens=depth))
  with patch.object(ModelRegistry,'resolve_model_cls',return_value=(Qwen4ExpForCausalLM,None)),patch('vllm.config.vllm.set_current_vllm_config',lambda _:contextlib.nullcontext()):
   Platform._align_hybrid_block_size(cfg,Backend)
  out.append(dict(mtp=depth,requested_block=block,requested_mamba_block=64,resolved_block=c.block_size,resolved_mamba_block=c.mamba_block_size,padded_page_bytes=c.mamba_page_size_padded))
# Exercise installed partial-tail method with a minimal block-pool spy.
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
calls=[]
pool=N(hash_block_size=64,cache_partial_block=lambda **kw: calls.append(kw['num_tokens']) or b'hash')
manager=N(block_size=1920,block_pool=pool,req_to_blocks={'r':[N(is_null=False) for _ in range(6)]},kv_cache_group_id=0,_partial_hit_reqs={},num_cached_block={},_producer_partial_tail_reqs={})
request=N(request_id='r',num_prompt_tokens=10001)
checks=[]
for n in [9984,10048,10112]:
 result=MambaManager._cache_partial_tail_block(manager,request,n)
 checks.append({'position':n,'registered':result is not None})
assert [x['registered'] for x in checks]==[True,False,False]
print('RESULT_JSON='+json.dumps({'alignment':out,'partial_tail':checks,'scope':'Installed Platform alignment method and MambaManager registration method; model shape calculators, no model weights or inference. Backend fixture uses MultipleOf(16), one BF16 KV head at actual model head dimension.'}))
