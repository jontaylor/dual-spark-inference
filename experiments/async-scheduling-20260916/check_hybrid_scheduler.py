"""Instantiate the complete candidate scheduler/connector with a tiny hybrid pool.

Runs actual CPU scheduling and connector planning, no model or disk worker.
"""
import importlib.util,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace as N
import torch
P=Path(sys.argv[1])
mode=sys.argv[2] if len(sys.argv)>2 else "normal"
assert mode in ("normal","cancel_save","cancel_restore","all_deferred")
for name,file in (
 ('vllm.distributed.kv_transfer.kv_connector.v1.gb10_completion','completion.py'),
 ('vllm.distributed.kv_transfer.kv_connector.v1.gb10_aligned_offloading_connector','connector.py'),
 ('vllm.v1.core.sched.gb10_parking_scheduler','parking_scheduler.py')):
 spec=importlib.util.spec_from_file_location(name,P/file)
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
from vllm.v1.core.sched.gb10_parking_scheduler import GB10AsyncParkingScheduler
from vllm.config import VllmConfig,ModelConfig,CacheConfig,SchedulerConfig,KVTransferConfig
from vllm.v1.kv_cache_interface import KVCacheConfig,KVCacheGroupSpec,FullAttentionSpec,MambaSpec,CircularBufferSpec,KVCacheTensor
from vllm.v1.outputs import ModelRunnerOutput,KVConnectorOutput
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingWorkerMetadata
from vllm.v1.request import Request,RequestStatus
from vllm import SamplingParams

model=Path('/tmp/async-hybrid-tiny-config');model.mkdir(exist_ok=True)
(model/'config.json').write_text(json.dumps(dict(architectures=['LlamaForCausalLM'],model_type='llama',
 hidden_size=64,intermediate_size=128,num_hidden_layers=1,num_attention_heads=4,
 num_key_value_heads=4,vocab_size=128,max_position_embeddings=1024)))
config=VllmConfig(model_config=ModelConfig(model=str(model),skip_tokenizer_init=True,
 dtype='bfloat16',max_model_len=1024,enforce_eager=True),
 cache_config=CacheConfig(block_size=64,enable_prefix_caching=False,mamba_cache_mode='align'),
 scheduler_config=SchedulerConfig(is_encoder_decoder=False,max_model_len=1024,max_num_seqs=4,
 max_num_batched_tokens=256,async_scheduling=True))
# No model is loaded. Supply the connector's Qwen4-specific geometry contract.
config.model_config.hf_text_config.model_type='qwen4_exp_text'
config.model_config.hf_text_config.indexer_compress_ratio=4
config.kv_transfer_config=KVTransferConfig(kv_connector='GB10AlignedOffloadingConnector',
 kv_connector_module_path='vllm.distributed.kv_transfer.kv_connector.v1.gb10_aligned_offloading_connector',
 kv_role='kv_both',kv_connector_extra_config=dict(
 spec_name='RankLocalDiskOffloadingSpec',spec_module_path='vllm.v1.kv_offload.rank_local_disk',
 disk_bytes_per_rank=2**20,staging_blocks=2,root_dir='/tmp/async-hybrid-unused-disk',
 blocks_per_chunk=1,offload_prompt_only=False,completion_checkpoints=True,
 disk_write_policy='pressure',memory_completion_cache=True,native_completion_cache=True,
 parking_reservation_tokens=64,async_terminal_snapshot_bytes=2**20))
config.cache_config.num_gpu_blocks=128
kv=KVCacheConfig(num_blocks=128,kv_cache_tensors=[KVCacheTensor(size=128*4096,layers=[name],layer_stride=4096,block_stride=4096) for name in ("att","rec","ring")],kv_cache_groups=[
 KVCacheGroupSpec(layer_names=['att'],kv_cache_spec=FullAttentionSpec(block_size=64,num_kv_heads=1,head_size=16,dtype=torch.bfloat16)),
 KVCacheGroupSpec(layer_names=['rec'],kv_cache_spec=MambaSpec(block_size=64,shapes=((6,16),(2,8,8)),
 dtypes=(torch.bfloat16,torch.float32),mamba_cache_mode='align',num_speculative_blocks=3,page_size_padded=4096)),
 KVCacheGroupSpec(layer_names=['ring'],kv_cache_spec=CircularBufferSpec(block_size=8,num_kv_heads=1,head_size=2,
 dtype=torch.bfloat16,page_size_padded=4096))])
s=GB10AsyncParkingScheduler(config,kv,N(should_advance=lambda *args,**kwargs:False),block_size=64,hash_block_size=64)
s.connector.connector_scheduler.config = s.connector.connector_scheduler.config._replace(num_workers=2)
if mode=='all_deferred':
 from vllm.v1.kv_offload.gb10_allocation_deferral import AllocationDeferred
 for i in range(4):
  s.add_request(Request(request_id=f'D{i}',prompt_token_ids=list(range(60)),
   sampling_params=SamplingParams(max_tokens=32),pooling_params=None))
 allocate=s.kv_cache_manager.allocate_slots
 def defer(*args,**kwargs):raise AllocationDeferred()
 s.kv_cache_manager.allocate_slots=defer
 one=s.schedule();two=s.schedule()
 assert one.total_num_scheduled_tokens==two.total_num_scheduled_tokens==0
 assert not one.preempted_req_ids and s.has_requests()
 assert one.kv_connector_metadata is not None and two.kv_connector_metadata is not None
 versions=dict(s.connector._terminal_versions)
 assert len(versions)==4
 s.kv_cache_manager.allocate_slots=allocate
 three=s.schedule()
 assert len(three.num_scheduled_tokens)==4
 assert three.kv_connector_metadata.terminal_versions==versions
 print(json.dumps(dict(passed=True,scenario=mode,checks=['all-deferred returns valid connector-only batches','bounded generations survive repeated deferral','admission recovers without changing generations'])))
 sys.exit(0)
a=Request(request_id='A',prompt_token_ids=list(range(60)),sampling_params=SamplingParams(max_tokens=32,stop_token_ids=[99]),pooling_params=None)
s.add_request(a)
one=s.schedule();two=s.schedule()
assert one.num_scheduled_tokens['A']==60 and two.num_scheduled_tokens['A']==1
assert one.kv_connector_metadata.terminal_versions['A']>0
assert len(s.connector._terminal_versions)==1
s.update_from_output(one,ModelRunnerOutput(req_ids=['A'],req_id_to_index={'A':0},sampled_token_ids=[[99]]))
assert a.is_finished() and a.num_in_flight_tokens==1
assert 'A' in s.connector._completion.pending or 'A' in s.connector._completion.deferred_allocations
assert 'A' in s.connector._terminal_versions
s.update_from_output(two,ModelRunnerOutput(req_ids=['A'],req_id_to_index={'A':0},sampled_token_ids=[[12]]))
assert list(a.output_token_ids)==[99] and a.num_in_flight_tokens==0
# Actual transfer completion accounting must retain the generation after the
# first rank and release it only after the second rank's completion.
version = s.connector.terminal_version('A')
transfer_step = s.schedule()
save = transfer_step.kv_connector_metadata.completion_saves['A']
assert save.terminal_version == version
assert not transfer_step.total_num_scheduled_tokens

def ack(step):
 s.update_from_output(step,ModelRunnerOutput(req_ids=[],req_id_to_index={},
  kv_connector_output=KVConnectorOutput(kv_connector_worker_meta=
   OffloadingWorkerMetadata(completed_jobs={save.job_id:1}))))
ack(transfer_step)
assert 'A' in s.connector._terminal_versions and 'A' in s.requests
assert 'A' in s.connector._completion.pending
empty=s.schedule();ack(empty)
assert 'A' not in s.connector._terminal_versions and 'A' not in s.requests
release=s.schedule()
assert release.kv_connector_metadata.terminal_releases == [('A',version)]
assert not s.schedule().kv_connector_metadata.terminal_releases
# Quiesce/drain an active request, finish its pressure checkpoint on both
# ranks, then exercise same-pass park/re-admission and restore planning.
p=Request(request_id='P',prompt_token_ids=list(range(60)),sampling_params=SamplingParams(max_tokens=64),pooling_params=None)
q=Request(request_id='Q',prompt_token_ids=list(range(60)),sampling_params=SamplingParams(max_tokens=64),pooling_params=None)
s.add_request(p);s.add_request(q);s._force_after=1
one=s.schedule();two=s.schedule()
def deliver(step, completed=None, recving=None):
 ids=list(step.num_scheduled_tokens)
 s.update_from_output(step,ModelRunnerOutput(req_ids=ids,req_id_to_index={rid:i for i,rid in enumerate(ids)},
  sampled_token_ids=[[7] for _ in ids],kv_connector_output=(KVConnectorOutput(
   finished_recving=recving,kv_connector_worker_meta=OffloadingWorkerMetadata(completed_jobs=completed or {})) if completed or recving else None)))
deliver(one)
three=s.schedule()
assert 'P' in s._quiescing and 'P' not in three.num_scheduled_tokens
assert not s.connector.pressure_save_started('P')
deliver(two)
assert p.num_in_flight_tokens==0
four=s.schedule()
assert s.connector.pressure_save_started('P')
pressure_save=four.kv_connector_metadata.completion_saves['P']
assert pressure_save.terminal_version is None
old_version=s.connector.terminal_version('P')
if mode=='cancel_save':
 s.finish_requests('P',RequestStatus.FINISHED_ABORTED)
 assert 'P' in s.requests and 'P' in s.connector._terminal_versions
 assert 'P' in s.connector._completion.pending
deliver(three)
five=s.schedule();deliver(four,{pressure_save.job_id:1})
assert 'P' not in s._parked
six=s.schedule();deliver(five,{pressure_save.job_id:1})
seven=s.schedule()
if mode=='cancel_save':
 assert 'P' not in s.requests and 'P' not in s.connector._terminal_versions
 assert 'P' not in s.connector._completion.pending
 assert ('P',old_version) in seven.kv_connector_metadata.terminal_releases
 assert 'P' not in s._quiescing and 'P' not in s.parking.tickets
 assert 'P' not in s.connector._pressure_saves
 print(json.dumps(dict(passed=True,scenario=mode,checks=['cancelled capture retains state until both completions','generation and parking reservations released after drain'])))
 sys.exit(0)
assert 'P' in seven.preempted_req_ids and 'P' in s._parked
assert ('P',old_version) in seven.kv_connector_metadata.terminal_releases
loads={jid:job for jid,job in seven.kv_connector_metadata.load_jobs.items() if job.req_id=='P'}
assert loads and not (set(loads)&seven.kv_connector_metadata.jobs_to_flush)
assert s.connector.terminal_version('P') != old_version
# Load acknowledgements arrive in two subsequent worker outputs. The request
# remains blocked after one; after both, the scheduler promotes and restores it.
deliver(six)
eight=s.schedule();deliver(seven,{jid:1 for jid in loads})
assert 'P' in s._parked
restore_version=s.connector.terminal_version('P')
if mode=='cancel_restore':
 s.finish_requests('P',RequestStatus.FINISHED_ABORTED)
 assert 'P' in s.requests and 'P' in s.connector._terminal_versions
nine=s.schedule();deliver(eight,{jid:1 for jid in loads},recving={'P'})
ten=s.schedule()
if mode=='cancel_restore':
 assert 'P' not in s.requests and 'P' not in s._parked
 assert 'P' not in s.connector._terminal_versions and 'P' not in s.parking.tickets
 assert ('P',restore_version) in ten.kv_connector_metadata.terminal_releases
 assert 'P' not in ten.num_scheduled_tokens
 print(json.dumps(dict(passed=True,scenario=mode,checks=['cancel during pending restore retains ownership','final restore completion frees without resuming','new generation released exactly once'])))
 sys.exit(0)
assert 'P' not in s._parked and 'P' in ten.num_scheduled_tokens
assert ten.kv_connector_metadata.terminal_versions['P'] != old_version
print(json.dumps(dict(passed=True,checks=['full candidate constructor and hybrid allocator',
 'two outstanding batches with real connector metadata','terminal snapshot planned while later batch in flight',
 'terminal generation retained pending copy','later output drains without append','both-rank completion gate releases generation exactly once','quiesce/drain pressure save','same-pass park/resume retains fresh load','both-rank restore creates new generation'])))
