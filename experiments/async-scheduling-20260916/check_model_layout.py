"""Validate candidate descriptors against source-derived model shapes on meta.

Uses the deployed Mamba/QSA binders with padded block strides. No weights,
CUDA allocations, or live-worker inspection. This is not a live tensor dump.
"""
import importlib.util,json,math,sys
from pathlib import Path
from types import SimpleNamespace as N
import torch
P=Path(sys.argv[1]);config=json.loads(Path(sys.argv[2]).read_text());c=config.get('text_config',config)
for name,file in (
 ('vllm.distributed.kv_transfer.kv_connector.v1.gb10_completion','completion.py'),
 ('vllm.v1.kv_offload.gb10_terminal_state','terminal_state.py'),
 ('vllm.v1.kv_offload.gb10_terminal_versions','terminal_versions.py'),
 ('vllm.v1.kv_offload.gb10_terminal_worker','terminal_worker.py')):
 spec=importlib.util.spec_from_file_location(name,P/file);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
from vllm.v1.kv_offload.gb10_terminal_worker import TerminalWorker
from vllm.v1.kv_cache_interface import MambaSpec,CircularBufferSpec,KVCacheGroupSpec
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator as Shapes,get_conv_copy_spec,get_temporal_copy_spec
from vllm.models.qwen4_exp.common.qsa_cache import QSAKeyStateCache
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum as Kind
nblocks=1581;page=27156480;context={};groups=[]
conv,ssm=Shapes.gated_delta_net_state_shape(2,c['linear_num_key_heads'],c['linear_num_value_heads'],c['linear_key_head_dim'],c['linear_value_head_dim'],c['linear_conv_kernel_dim'],3)
ple=Shapes.short_conv_state_shape(1,c['hidden_size']*c['hc_count'],(c['ple_conv_kernel_size']-1)*c['ngram_size']+1,3)[0]
# Source-derived layer inventory; grouping only exercises per-group indexing.
for gi,count in enumerate((13,13,10,1)):
 shapes,dtypes,kind=((conv,ssm),(torch.bfloat16,torch.float32),Kind.GDN_ATTN) if gi<3 else ((ple,),(torch.bfloat16,),Kind.SHORT_CONV)
 names=[]
 for li in range(count):
  name=f'g{gi}.{li}';names.append(name)
  content=sum(math.prod(shape)*torch.empty((),dtype=dtype).element_size() for shape,dtype in zip(shapes,dtypes))
  raw=torch.empty_strided((nblocks,1,1,content),(page,page,page,1),dtype=torch.int8,device='meta')
  layer=N(get_state_shape=lambda shapes=shapes: shapes,get_state_dtype=lambda dtypes=dtypes:dtypes)
  MambaBase.bind_kv_cache(layer,raw);context[name]=layer
 groups.append(KVCacheGroupSpec(layer_names=names,kv_cache_spec=MambaSpec(block_size=1920,shapes=shapes,dtypes=dtypes,mamba_type=kind,mamba_cache_mode='align',num_speculative_blocks=3)))
names=[]
for li in range(13):
 name=f'ring.{li}';names.append(name)
 layer=QSAKeyStateCache.__new__(QSAKeyStateCache);torch.nn.Module.__init__(layer)
 layer.head_size=140;layer.key_head_size=128;layer.cache_rope_positions=True;layer.rope_position_offset=128
 raw=torch.empty_strided((nblocks,1,8,140),(page//2,1120,140,1),dtype=torch.bfloat16,device='meta')
 layer.bind_kv_cache(raw);context[name]=layer
 assert layer.rope_position_cache.shape[-1]==3
 groups.append(KVCacheGroupSpec(layer_names=[name],kv_cache_spec=CircularBufferSpec(block_size=8,num_kv_heads=1,head_size=140,head_size_v=0,dtype=torch.bfloat16)))
funcs={Kind.GDN_ATTN:(get_conv_copy_spec,get_temporal_copy_spec),Kind.SHORT_CONV:(get_conv_copy_spec,)}
runner=N(max_num_reqs=32,kv_cache_config=N(num_blocks=nblocks),device=torch.device('meta'),
 vllm_config=N(compilation_config=N(static_forward_context=context),kv_transfer_config=N(kv_connector_extra_config={'async_terminal_snapshot_bytes':2**31})),
 model=N(get_mamba_state_copy_funcs=lambda kinds:funcs),block_tables=N(block_tables=[N(gpu=torch.empty((32,8),dtype=torch.int32,device='meta')) for _ in groups]))
worker=TerminalWorker(N(_completion_groups=groups,_completion_order=tuple(range(len(groups)))),runner)
assert len(worker.pieces)==86 and worker.byte_count==59109824
assert worker.payload_data.numel()==32*59109824
print(json.dumps(dict(passed=True,scope='deployed binders and reconstructed shapes on meta, not live-worker inventory',
 pieces=len(worker.pieces),bytes_per_request=worker.byte_count,payload_bytes=worker.payload_data.numel(),checks=['padded Mamba binder views','QSA ring with int64 MRoPE tail','candidate contiguous-block assumptions','full source-derived content size'])) )
