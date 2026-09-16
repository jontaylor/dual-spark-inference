"""Real async scheduling/accounting with parking policy; no model or transfer I/O.

The connector facade deliberately excludes checkpoint transport, covered
separately. This checks two outstanding batches and policy quiescence using
upstream scheduling/output methods, not a hand-written token counter.
"""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as N
import torch
P=Path(sys.argv[1])
for name,file in (
 ('vllm.distributed.kv_transfer.kv_connector.v1.gb10_completion','completion.py'),
 ('vllm.distributed.kv_transfer.kv_connector.v1.gb10_aligned_offloading_connector','connector.py'),
 ('vllm.v1.core.sched.gb10_parking_scheduler','parking_scheduler.py')):
 spec=importlib.util.spec_from_file_location(name,P/file)
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
from vllm.v1.core.sched.gb10_parking_scheduler import GB10AsyncParkingScheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.parking_policy import ParkingPolicy
from vllm.config import VllmConfig,ModelConfig,CacheConfig,SchedulerConfig
from vllm.v1.kv_cache_interface import KVCacheConfig,KVCacheGroupSpec,FullAttentionSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm import SamplingParams

model=Path('/tmp/async-scheduler-tiny-config');model.mkdir(exist_ok=True)
(model/'config.json').write_text(json.dumps(dict(architectures=['LlamaForCausalLM'],model_type='llama',
 hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,
 num_key_value_heads=4,vocab_size=128,max_position_embeddings=256)))
config=VllmConfig(model_config=ModelConfig(model=str(model),skip_tokenizer_init=True,
 dtype='bfloat16',max_model_len=256,enforce_eager=True),
 cache_config=CacheConfig(block_size=16,enable_prefix_caching=False),
 scheduler_config=SchedulerConfig(is_encoder_decoder=False,max_model_len=256,max_num_seqs=4,
 max_num_batched_tokens=256,async_scheduling=True))
config.cache_config.num_gpu_blocks=64
kv=KVCacheConfig(num_blocks=64,kv_cache_tensors=[],kv_cache_groups=[KVCacheGroupSpec(
 layer_names=['layer'],kv_cache_spec=FullAttentionSpec(block_size=16,num_kv_heads=4,
 head_size=16,dtype=torch.bfloat16))])

class Fixture(GB10AsyncParkingScheduler):
 @property
 def _parking_connector(self):return self.transport

def fixture():
 s=object.__new__(Fixture)
 AsyncScheduler.__init__(s,config,kv,N(should_advance=lambda *args, **kwargs: False),block_size=16,hash_block_size=16)
 s.parking=ParkingPolicy(63,unit=64,unit_cost=4,request_cost=1)
 s.transport=N(retry_completion_allocations=lambda:None,can_reserve_terminal=lambda rid:True,
  reserve_terminal=lambda rid:True,pressure_save_started=lambda rid:False,
  release_terminal=lambda rid:None,_native_pressure=None)
 s._quiescing={};s._parked={};s._generations={};s._park_commit=set()
 s._force_after=0;s._forced=False;s._forced_request=None;s._pressure_only=False
 s._memory_completions=False;s._save_timeout=30
 return s

def request(rid,limit=16):
 return Request(request_id=rid,prompt_token_ids=[1,2,3,4],
  sampling_params=SamplingParams(max_tokens=limit),pooling_params=None)

def deliver(s,step,tokens):
 ids=list(step.num_scheduled_tokens)
 return s.update_from_output(step,ModelRunnerOutput(req_ids=ids,
  req_id_to_index={rid:i for i,rid in enumerate(ids)},sampled_token_ids=[tokens[rid] for rid in ids]))

s=fixture();a=request('A');b=request('B');s.add_request(a);s.add_request(b)
first=s.schedule();second=s.schedule()
assert set(first.num_scheduled_tokens)=={'A','B'}
assert set(second.num_scheduled_tokens)=={'A','B'}
assert a.num_in_flight_tokens==5 and a.num_output_placeholders==2
# Force quiescence while two batches remain outstanding. Neither may be parked.
s._forced_request='A';s._quiescing['A']=__import__('time').monotonic()
assert not s._try_park(a)
third=s.schedule()
assert 'A' not in third.num_scheduled_tokens and 'B' in third.num_scheduled_tokens
assert a in s.running
# Drain one output: the later batch still owns tokens and prevents parking.
deliver(s,first,{'A':[7],'B':[8]})
assert a.num_in_flight_tokens==1 and a.num_output_placeholders==1
assert not s._try_park(a)
deliver(s,second,{'A':[9],'B':[10]})
assert a.num_in_flight_tokens==0 and a.num_output_placeholders==0
assert list(a.output_token_ids)==[7,9]
# Terminal first output while another batch is outstanding: later output cannot
# append another token to the finished request.
s=fixture();a=request('T');a.sampling_params.stop_token_ids=[11];s.add_request(a)
first=s.schedule();second=s.schedule()
assert a.num_output_placeholders==2
deliver(s,first,{'T':[11]})
assert a.is_finished() and list(a.output_token_ids)==[11]
deliver(s,second,{'T':[12]})
assert list(a.output_token_ids)==[11]
# Three speculative placeholders per step: reject two drafts in the first
# decode output while the next decode is already queued.
s=fixture();s.num_spec_tokens=3;s.num_lookahead_tokens=3
s._spec_token_placeholders=[-1]*3
a=request('S',32);s.add_request(a)
prefill=s.schedule();deliver(s,prefill,{'S':[6]})
first=s.schedule();second=s.schedule()
assert len(first.scheduled_spec_decode_tokens['S'])==3
assert a.num_output_placeholders==8 and a.num_in_flight_tokens==8
deliver(s,first,{'S':[7,8]})
assert a.num_output_placeholders==4 and a.num_in_flight_tokens==4
deliver(s,second,{'S':[9,10,11,12]})
assert a.num_output_placeholders==0 and a.num_in_flight_tokens==0
assert list(a.output_token_ids)==[6,7,8,9,10,11,12]
# Draining/rejecting two queued batches can shrink the placeholder-inclusive
# token footprint. The parking reservation is a high-water mark, not a demand
# that accepted token counts never fall below speculative reservations.
s=fixture();s.num_spec_tokens=3;s.num_lookahead_tokens=3;s._spec_token_placeholders=[-1]*3
a=request('R',32);s.add_request(a)
prefill=s.schedule();deliver(s,prefill,{'R':[6]})
first=s.schedule();second=s.schedule()
peak=s.parking.tickets['R'].tokens
deliver(s,first,{'R':[7]});deliver(s,second,{'R':[8]})
assert a.num_tokens+a.num_output_placeholders+s.num_lookahead_tokens+1 < peak
s.schedule()
assert s.parking.tickets['R'].tokens >= peak
print(json.dumps(dict(passed=True,checks=['two real outstanding batches',
 'quiescing request held while peers advance','in-flight work prevents parking',
 'output delivery retires placeholders','terminal later output ignored','three-draft rejection with later batch outstanding','speculative drain retains reservation high-water mark']),sort_keys=True))
