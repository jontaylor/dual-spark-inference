"""Run the actual candidate scheduler without loading any model weights."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as N
import torch

P = Path(__file__).parent
def load(name, file):
    spec = importlib.util.spec_from_file_location(name, P / file)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

load('vllm.v1.kv_offload.gb10_allocation_deferral', 'deferral.py')
load('vllm.v1.core.block_pool', 'block_pool.py')
load('vllm.v1.core.kv_cache_manager', 'kv_cache_manager.py')
sched = load('vllm.v1.core.sched.scheduler', 'scheduler.py')
from vllm.config import VllmConfig, ModelConfig, CacheConfig, SchedulerConfig
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, FullAttentionSpec
from vllm.v1.request import Request
from vllm import SamplingParams
from vllm.v1.kv_offload.gb10_allocation_deferral import AllocationDeferred

model = P / 'tiny-config'; model.mkdir(exist_ok=True)
(model/'config.json').write_text(json.dumps(dict(architectures=['LlamaForCausalLM'],
    model_type='llama', hidden_size=64, intermediate_size=128, num_hidden_layers=1,
    num_attention_heads=4, num_key_value_heads=4, vocab_size=128, max_position_embeddings=256)))
config = VllmConfig(model_config=ModelConfig(model=str(model), skip_tokenizer_init=True,
    dtype='bfloat16', max_model_len=256, enforce_eager=True),
    cache_config=CacheConfig(block_size=16, enable_prefix_caching=False),
    scheduler_config=SchedulerConfig(is_encoder_decoder=False, max_model_len=256, max_num_seqs=8,
                                    max_num_batched_tokens=256, async_scheduling=False))
config.cache_config.num_gpu_blocks = 64
kv = KVCacheConfig(num_blocks=64, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(
    layer_names=['layer'], kv_cache_spec=FullAttentionSpec(block_size=16,
    num_kv_heads=4, head_size=16, dtype=torch.bfloat16))])

def fixture():
    return sched.Scheduler(config, kv, N(), block_size=16, hash_block_size=16)

def req(rid):
    return Request(request_id=rid, prompt_token_ids=[1,2,3,4], sampling_params=SamplingParams(max_tokens=16),
                   pooling_params=None)

s = fixture(); a, b = req('A'), req('B')
s.add_request(a); s.add_request(b)
real = s.kv_cache_manager.allocate_slots
def alloc(request, *args, **kwargs):
    if request.request_id == 'A': raise AllocationDeferred()
    return real(request, *args, **kwargs)
s.kv_cache_manager.allocate_slots = alloc
out = s.schedule()
assert 'A' not in out.num_scheduled_tokens and 'B' in out.num_scheduled_tokens
assert a.num_computed_tokens == 0

# Admit A, then test actual running loop: one request defers, another runs.
s.kv_cache_manager.allocate_slots = real
a2 = req('C'); s.add_request(a2)
out = s.schedule()
assert 'A' in out.num_scheduled_tokens and 'C' in out.num_scheduled_tokens
for r in (a, a2):
    r.num_in_flight_tokens = 0
    r.append_output_token_ids([7])
s.kv_cache_manager.allocate_slots = alloc
out = s.schedule()
assert 'A' not in out.num_scheduled_tokens and 'C' in out.num_scheduled_tokens
assert not out.preempted_req_ids

# All runnable requests deferred: scheduler returns a valid empty model batch.
def defer_all(*args, **kwargs): raise AllocationDeferred()
s.kv_cache_manager.allocate_slots = defer_all
a.append_output_token_ids([8]); a2.append_output_token_ids([8])
out = s.schedule()
assert out.total_num_scheduled_tokens == 0 and not out.preempted_req_ids
assert s.has_requests()
print(json.dumps({'passed': True, 'checks': ['waiting A deferred while B scheduled',
    'running A deferred while C scheduled', 'no victim preemption',
    'all deferred yields empty model batch and engine stays alive']}))
