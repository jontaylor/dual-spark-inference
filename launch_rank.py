#!/usr/bin/env python3
"""Launch one rank using pinned Qwen FP8 weights and local immutable overlays."""
import json
from runtime_config import load_config, model_snapshot, resolve_path
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
cfg = load_config(root)
rank = int(sys.argv[1])
assert rank in (0,1)
assert cfg['model_id'] == 'Qwen/Qwen3.8-Flash-Next-FP8'
assert cfg['kv_cache_dtype'] == 'bfloat16'
assert cfg['tensor_parallel_size'] == 2 and cfg['expert_parallel'] and cfg['mtp_tokens'] == 3
snapshot = (resolve_path(cfg['paths']['hf_cache'], root) / ('models--' + cfg['model_id'].replace('/', '--')) / 'snapshots')/cfg['revision']
source_config = json.loads((snapshot/'config.json').read_text())
assert source_config['quantization_config']['quant_method'] == 'fp8'
assert not (snapshot/'hf_quant_config.json').exists(), 'Unexpected ModelOpt sidecar'
manifest = json.loads((root/'verified-checkpoint.json').read_text())
assert manifest['revision'] == cfg['revision'] and manifest['model_id'] == cfg['model_id']
for f in manifest['files']:
    if (snapshot/f['name']).stat().st_size != f['size']:
        raise RuntimeError(f'Checkpoint changed since verification: {f["name"]}')
name = 'qwen38-next-qwen-fp8-r'+str(rank)
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
overlays = json.loads((root/'server/overlays.json').read_text())
cmd=['docker','run','--name',name,'--gpus','all','--network','host','--ipc','host',
     '--cap-add','SYS_NICE','--ulimit','memlock=-1','--ulimit','stack=67108864',
     '--device','/dev/infiniband:/dev/infiniband','--entrypoint','python3']
def mount(src,dest):
    assert Path(src).exists(), src
    cmd.extend(['-v',f'{src}:{dest}:ro'])
model_path = '/model-store/snapshots/' + cfg['revision']
mount(snapshot.parent.parent,'/model-store')
mount(root/'container_entry.py','/opt/container_entry.py')
if not cfg.get('server_in_image'):
    mount(root/'files/gb10','/opt/gb10')
mount(packed_dir,'/ple-cache')
if not cfg.get('server_in_image'):
    for file,dest in overlays.items():mount(root/'files'/file,f'{pkg}/{dest}')
for src,dest in [('config_patched.json','config.json'),('hf_quant_config_patched.json','hf_quant_config.json')]:
    if (root/'files'/src).exists():mount(root/'files'/src,model_path+'/'+dest)
if rank==0:mount(resolve_path(cfg['api_key_file'], root),'/run/secrets/inference-api-key')
cache=resolve_path(cfg['paths']['runtime_cache'], root);cache.mkdir(parents=True,exist_ok=True)
cmd.extend(['-v',f'{cache}:/root/.cache/vllm'])
env={
 'GLOO_SOCKET_IFNAME':cfg['interface'],'NCCL_SOCKET_IFNAME':cfg['interface'],
 'TP_SOCKET_IFNAME':cfg['interface'],'NCCL_IB_DISABLE':'0','NCCL_IB_HCA':cfg['ib_hca'],
 'NCCL_IB_GID_INDEX':'3','NCCL_IB_AUTO_DETECT':'0','NCCL_DEBUG':'WARN',
 'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','VLLM_PLE_PACKED_TABLE_DIR':'/ple-cache',
 'VLLM_PLE_CPU_OFFLOAD':'1','VLLM_PLE_LOCAL_TP':'1',
 'VLLM_HOST_IP':cfg['head_ip'] if rank==0 else cfg['worker_ip'],
 'VLLM_ENGINE_READY_TIMEOUT_S':'3600','VLLM_ALLOW_LONG_MAX_MODEL_LEN':'1',
}
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
tc={'ple_embedding_dtype':'float8_e4m3fn'}
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
        'cudagraph_capture_sizes':[4*s for s in range(1,cfg['max_num_seqs']+1)]})]
else:raise ValueError('Unknown graph mode')
if rank:args+=['--headless']
else:args+=['--host','0.0.0.0','--port',str(cfg['port'])]
cmd += [cfg['image'],'/opt/container_entry.py',*args]
print(json.dumps({'rank':rank,'image':cfg['image'],'model':cfg['model_id'],'revision':cfg['revision'],
                  'port':cfg['port'],'ple':'local_nvme_mmap','args':args}),flush=True)
if '--dry-run' in sys.argv:
    print(json.dumps({'docker_command': cmd}), flush=True)
else:
    sys.exit(subprocess.call(cmd))
