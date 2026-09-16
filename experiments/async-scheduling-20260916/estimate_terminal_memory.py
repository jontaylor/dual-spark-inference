"""Config/source-derived packed-state estimate; not a live tensor inspection."""
import json
import math
from pathlib import Path
import sys
import torch
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator as Shapes
cfg=json.loads(Path(sys.argv[1]).read_text())
c=cfg.get('text_config',cfg)
tp=2;spec=3;capacity=32
conv,ssm=Shapes.gated_delta_net_state_shape(tp,c['linear_num_key_heads'],
 c['linear_num_value_heads'],c['linear_key_head_dim'],c['linear_value_head_dim'],
 c['linear_conv_kernel_dim'],spec)
gdn_count=c['layer_types'].count('linear_attention')
gdn=gdn_count*(math.prod(conv)*2+math.prod(ssm)*4)
ple_shape=Shapes.short_conv_state_shape(1,c['hidden_size']*c['hc_count'],
 (c['ple_conv_kernel_size']-1)*c['ngram_size']+1,spec)[0]
ple=len(c['ple_layer_ids'])*math.prod(ple_shape)*2
# Each full-attention indexer, including the full-attention MTP layer,
# owns a raw-key ring. MRoPE stores three int64 coordinates as 12 BF16 cells.
ring_count=c['layer_types'].count('full_attention')+c['mtp']['layer_types'].count('full_attention')
ratio=c['indexer_compress_ratio'];ring_capacity=ratio*math.ceil((ratio+spec)/ratio)
width=4*math.ceil(c['indexer_head_dim']/4)+12
rings=ring_count*ring_capacity*width*2
per_request=gdn+ple+rings
print(json.dumps(dict(scope='config/source-derived estimate; validate against bound worker tensors before deployment',
 tp=tp,speculative_tokens=spec,capacity=capacity,gdn_layers=gdn_count,gdn_shapes=[conv,ssm],
 ple_shape=ple_shape,ring_layers=ring_count,ring_shape=[ring_capacity,1,width],
 gdn_bytes=gdn,ple_bytes=ple,ring_bytes=rings,packed_bytes_per_request=per_request,
 packed_capacity_bytes=capacity*per_request,packed_capacity_gib=capacity*per_request/2**30,
 proposed_reservation_bytes=2**31,remaining_kv_pool_bytes=40*2**30-2**31),indent=2))
