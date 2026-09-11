#!/usr/bin/env python3
"""Launch a pinned NVIDIA NVFP4 rank with the same-process request pager."""
import json
from runtime_config import load_config, model_snapshot, resolve_path
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
cfg = load_config(root)
rank = int(sys.argv[1])
assert rank in (0,1)
assert cfg['model_id'] == 'nvidia/Qwen3.8-Flash-Next-NVFP4'
assert cfg['kv_cache_dtype'] == 'bfloat16'
assert cfg['tensor_parallel_size'] == 2 and cfg['expert_parallel'] and cfg['mtp_tokens'] == 3
snapshot = (resolve_path(cfg['paths']['hf_cache'], root) / ('models--' + cfg['model_id'].replace('/', '--')) / 'snapshots')/cfg['revision']
source_config = json.loads((snapshot/'config.json').read_text())
assert source_config['quantization_config']['quant_method'] == 'modelopt'
assert any(x.get('quant_algo') == 'NVFP4' for x in source_config['quantization_config']['quantized_layers'].values())
manifest = json.loads((Path(__file__).resolve().parent/'verified-checkpoint.json').read_text())
assert manifest['revision'] == cfg['revision'] and manifest['model_id'] == cfg['model_id']
for f in manifest['files']:
    if (snapshot/f['name']).stat().st_size != f['size']:
        raise RuntimeError(f'Checkpoint changed since verification: {f["name"]}')
name = 'qwen38-kv-paging-r'+str(rank)
packed_dir=resolve_path(cfg['paths']['ple_cache'], root)/cfg['revision']
metadata_files=list(packed_dir.glob('*.packed_u8.json'))
if len(metadata_files)!=1:
    raise RuntimeError('Expected exactly one verified FP8 packed PLE table')
for metadata_file in metadata_files:
    metadata=json.loads(metadata_file.read_text())
    if metadata['snapshot']!=cfg['revision'] or metadata['row_format']!='fp8_e4m3':
        raise RuntimeError('Packed PLE revision/format mismatch')
    if metadata_file.with_suffix('').stat().st_size!=metadata['total_rows']*metadata['row_width']:
        raise RuntimeError('Truncated packed PLE table')
existing = subprocess.run(['docker','inspect','--format','{{.State.Running}}',name],capture_output=True,text=True)
if existing.returncode == 0 and '--dry-run' not in sys.argv:
    if existing.stdout.strip() == 'true':
        raise RuntimeError('Rank container already running')
    subprocess.run(['docker','rm',name],check=True)
pkg = '/usr/local/lib/python3.12/dist-packages/vllm'
overlays = {} if cfg.get('server_in_image') else json.loads((root/'server/overlays.json').read_text())
from paging_ops import verify_runtime
verify_runtime(cfg, root, rank)
cmd=['docker','run','--name',name,'--gpus','all','--network','host','--ipc','host',
     '--cap-add','SYS_NICE','--ulimit','memlock=-1','--ulimit','stack=67108864',
     '--device','/dev/infiniband:/dev/infiniband','--entrypoint','python3']
def mount(src,dest):
    assert Path(src).exists(), src
    cmd.extend(['-v',f'{src}:{dest}:ro'])
model_path = '/model-store/snapshots/' + cfg['revision']
mount(snapshot.parent.parent,'/model-store')
mount(root/'files/paging/rank_local_disk.py', f'{pkg}/v1/kv_offload/rank_local_disk.py')
mount(root/'container_entry.py','/opt/container_entry.py')
mount(root/'files/paging/aligned_connector.py', f'{pkg}/distributed/kv_transfer/kv_connector/v1/gb10_aligned_offloading_connector.py')
cmd.extend(['-v', str(resolve_path(cfg['kv_paging']['disk_root'], root))+':/kv-disk'])
mount(Path(__file__).resolve().parent/'files/modelopt_patched.py', f'{pkg}/model_executor/layers/quantization/modelopt.py')
if not cfg.get('server_in_image'):
    mount(root/'files/gb10','/opt/gb10')
mount(packed_dir,'/ple-cache')
if not cfg.get('server_in_image'):
    for file,dest in overlays.items():mount(root/'files'/file,f'{pkg}/{dest}')
for src,dest in [('config_patched.json','config.json'),('hf_quant_config_patched.json','hf_quant_config.json')]:
    if (Path(__file__).resolve().parent/'files'/src).exists():mount(Path(__file__).resolve().parent/'files'/src,model_path+'/'+dest)
if rank==0:mount(resolve_path(cfg['api_key_file'], root),'/run/secrets/inference-api-key')
cache=resolve_path(cfg['paths']['runtime_cache'], root);cache.mkdir(parents=True,exist_ok=True)
cmd.extend(['-v',f'{cache}:/root/.cache/vllm'])
env={
 'GLOO_SOCKET_IFNAME':cfg['interface'],'NCCL_SOCKET_IFNAME':cfg['interface'],
 'TP_SOCKET_IFNAME':cfg['interface'],'NCCL_IB_DISABLE':'0','NCCL_IB_HCA':cfg['ib_hca'],
 'NCCL_IB_GID_INDEX':'3','NCCL_IB_AUTO_DETECT':'0','NCCL_DEBUG':'WARN',
 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','VLLM_PLE_PACKED_TABLE_DIR':'/ple-cache',
 'VLLM_PLE_CPU_OFFLOAD':'1','VLLM_PLE_LOCAL_TP':'1',
 'VLLM_USE_V2_MODEL_RUNNER':'1',
 'VLLM_HOST_IP':cfg['head_ip'] if rank==0 else cfg['worker_ip'],
 'VLLM_ENGINE_READY_TIMEOUT_S':'3600','VLLM_ALLOW_LONG_MAX_MODEL_LEN':'1',
}
if 'enable_roce_allreduce' in cfg:
    assert isinstance(cfg['enable_roce_allreduce'], bool)
    env['VLLM_ENABLE_ROCE_ALLREDUCE'] = str(int(cfg['enable_roce_allreduce']))
opts = cfg.get('optimizations', {})
if opts.get('kv_cache_accounting'):
    env['GB10_KV_ACCOUNTING'] = '1'
if opts.get('fair_prefill'):
    env['GB10_FAIR_PREFILL'] = '1'
if opts.get('bf16_kernels'):
    env['GB10_BF16_PLANS'] = '/opt/gb10/bf16_plans.json'
if opts.get('moe_sparse_activation'):
    env['GB10_MOE_SPARSE_ACTIVATION'] = '1'
if opts.get('moe_up_tensor'):
    env['GB10_MOE_UP_TENSOR'] = '1'
if opts.get('moe_down_tensor'):
    env['GB10_MOE_DOWN_TENSOR'] = '1'
if opts.get('draft_head_gemm'):
    env['GB10_DRAFT_HEAD_GEMM'] = '1'
if opts.get('ple_mapped_transport'):
    env['GB10_PLE_MAPPED_TRANSPORT'] = '1'
if opts.get('ple_native_hash'):
    env['GB10_PLE_NATIVE_HASH'] = '1'
env['GB10_PLE_ROW_CACHE_MB'] = str(opts.get('ple_row_cache_mb', 0))
env['GB10_PLE_GATHER_THREADS'] = str(opts.get('ple_gather_threads', 0))
if opts.get('ple_gather_threads', 0):
    env['OMP_WAIT_POLICY'] = 'PASSIVE'
if opts.get('draft_vocab_file'):
    mount(resolve_path(opts['draft_vocab_file'], root), '/opt/draft_vocab.txt')
    env['VLLM_MTP_DRAFT_VOCAB'] = '/opt/draft_vocab.txt'
if cfg.get('profiling', {}).get('enabled'):
    profiling = cfg['profiling']
    mount(resolve_path(profiling['nsys_root'], root), '/opt/nsight')
    profile_dir = resolve_path(profiling['output_root'], root) / ('rank' + str(rank))
    profile_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    cmd.extend(['-v', f'{profile_dir}:/profiles'])
    env['QWEN_NSYS_CAPTURE'] = '1'
if cfg.get('routing_capture'):
    routing = cfg['routing_capture']
    source = resolve_path(routing['source_root'], root)
    for relative in routing['files']:
        mount(source / relative, f'{pkg}/{relative}')
    output = resolve_path(routing['output_root'], root)
    output.mkdir(parents=True, mode=0o700, exist_ok=True)
    cmd.extend(['-v', f'{output}:/routing-capture'])
    env['GB10_ROUTING_RECORD_DIR'] = '/routing-capture'
for k,v in env.items():cmd.extend(['-e',f'{k}={v}'])
args=[model_path,'--served-model-name',*cfg['served_names'],
 '--tensor-parallel-size','2','--nnodes','2','--node-rank',str(rank),
 '--master-addr',cfg['head_ip'],'--master-port',str(cfg['master_port']),
 '--distributed-executor-backend','mp','--enable-expert-parallel','--all2all-backend','allgather_reducescatter',
 '--gpu-memory-utilization',str(cfg['gpu_memory_utilization']),
 '--max-num-seqs',str(cfg['max_num_seqs']),'--max-num-batched-tokens',str(cfg['max_num_batched_tokens']),
 '--max-model-len',str(cfg['max_model_len']),'--kv-cache-dtype',cfg['kv_cache_dtype'],
 '--load-format','safetensors','--safetensors-load-strategy','lazy','--enable-chunked-prefill',
 '--reasoning-parser','qwen3','--enable-auto-tool-choice','--tool-call-parser','qwen3_coder',
 '--mm-encoder-tp-mode',cfg['mm_encoder_tp_mode'],
 '--speculative-config',json.dumps({'method':'mtp','num_speculative_tokens':cfg['mtp_tokens'],
     'use_local_argmax_reduction':bool(opts.get('draft_local_argmax', False))})]
# Explicit policies keep changed defaults from altering this A/B test.
if cfg.get('mamba_cache_mode'):
    args += ['--mamba-cache-mode', cfg['mamba_cache_mode']]
if 'prefix_cache_retention_interval' in cfg:
    interval = cfg['prefix_cache_retention_interval']
    args += ['--prefix-cache-retention-interval', 'None' if interval is None else str(interval)]
if cfg.get('per_request_spec_decode_metrics'):
    args += ['--per-request-spec-decode-metrics', cfg['per_request_spec_decode_metrics']]
tc={'ple_embedding_dtype':'float8_e4m3fn'}
if cfg.get('enable_prompt_tokens_details', False):
    args += ['--enable-prompt-tokens-details']
if cfg['yarn_factor'] is not None:
    tc['rope_parameters']={'rope_type':'yarn','factor':cfg['yarn_factor'],'original_max_position_embeddings':262144}
args+=['--hf-overrides',json.dumps({'text_config':tc})]
if cfg.get('profiling', {}).get('enabled'):
    args += ['--profiler-config', json.dumps({
        'profiler': 'cuda', 'max_iterations': cfg['profiling']['max_iterations'],
    })]
if cfg['graph_mode']=='eager':args+=['--enforce-eager']
elif cfg['graph_mode']=='full_decode':
    args+=['--compilation-config',json.dumps({'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY',
        'cudagraph_capture_sizes':cfg.get('cudagraph_capture_sizes', [4*s for s in range(1,cfg['max_num_seqs']+1)])})]
else:raise ValueError('Unknown graph mode')
if rank:args+=['--headless']
else:args+=['--host','0.0.0.0','--port',str(cfg['port'])]
paging = cfg.get('kv_paging', {})
args += ['--kernel-config', '{"enable_flashinfer_autotune":false}', '--kv-cache-memory', str(paging.get('kv_bytes_per_rank', 4294967296)), '--kv-transfer-config', json.dumps({
 'kv_connector':'GB10AlignedOffloadingConnector',
 'kv_connector_module_path':'vllm.distributed.kv_transfer.kv_connector.v1.gb10_aligned_offloading_connector',
 'kv_role':'kv_both','kv_connector_extra_config':{
  'spec_name':'RankLocalDiskOffloadingSpec',
  'spec_module_path':'vllm.v1.kv_offload.rank_local_disk',
  'disk_bytes_per_rank':paging['disk_bytes_per_rank'],'staging_blocks':paging.get('staging_blocks',4),'root_dir':'/kv-disk','verify_transfers':paging.get('verify_transfers',True),
  **({'parking_block_budget': paging['block_budget']} if 'block_budget' in paging else {}),
  'parking_test_after_generated_tokens':paging.get('force_after_generated', 0),
  'blocks_per_chunk':1,'offload_prompt_only':False}})]
args += ['--no-async-scheduling', '--scheduler-cls', 'vllm.v1.core.sched.gb10_parking_scheduler.GB10ParkingScheduler']
for filename, target in [('parking_scheduler_base.py','scheduler.py'),('parking_policy.py','parking_policy.py'),('gb10_parking_scheduler.py','gb10_parking_scheduler.py')]:
    mount(root/'files/paging'/filename, f'{pkg}/v1/core/sched/{target}')
cmd += [cfg['image'],'/opt/container_entry.py',*args]
print(json.dumps({'rank':rank,'image':cfg['image'],'model':cfg['model_id'],'revision':cfg['revision'],
                  'port':cfg['port'],'ple':'local_nvme_mmap','args':args}),flush=True)
if '--dry-run' in sys.argv:
    print(json.dumps({'docker_command': cmd}), flush=True)
else:
    sys.exit(subprocess.call(cmd))
