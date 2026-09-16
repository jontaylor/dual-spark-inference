"""Isolated GPU validation/benchmark for the experimental GPU-hash connector."""
import argparse,json,pathlib,subprocess
p=argparse.ArgumentParser();p.add_argument('--benchmark',action='store_true');p.add_argument('--overlap',action='store_true');p.add_argument('--telemetry',action='store_true');p.add_argument('--async-ab',action='store_true');p.add_argument('--gate-timing',action='store_true');p.add_argument('--native-variant',choices=['14-batch-hash-async','15-sqpoll','16-sqpoll-ab','17-fixed-policy']);p.add_argument('--connector-variant',choices=['17-fixed-policy']);p.add_argument('--submission-policy',choices=['normal','sqpoll'],default='normal');args=p.parse_args()
b=pathlib.Path(__file__).resolve().parent;r=b.parents[1];old=r/'experiments/ple-in-process-20260916';v=pathlib.Path('/home/jon/vllm-gb10-kv-paging');candidate=b/('variants/'+args.connector_variant if args.connector_variant else 'variants/12-gpu-gate-timing' if args.gate_timing else 'variants/08-async-overlap' if args.async_ab else 'variants/06-overlap' if args.overlap else 'variants/03-gpu-hash');pkg='/usr/local/lib/python3.12/dist-packages/vllm'
cmd=['docker','run','--rm','--gpus','all','--network','none','--ipc','host','--ulimit','memlock=-1','--security-opt',f'seccomp={old}/seccomp-uring.json','-e','GB10_PLE_IN_PROCESS=1','-e','GB10_PLE_MAPPED_TRANSPORT=1','-e','VLLM_PLE_CPU_OFFLOAD=1','-e','VLLM_PLE_LOCAL_TP=1','-e','GB10_PLE_ROW_CACHE_MB=1','-e','VLLM_PLE_PACKED_TABLE_DIR=/tmp/ple-test-table','-e','GB10_PLE_GPU_HASH=1','-v',f'{candidate}:/build']
mounts={pkg+'/'+key:pathlib.Path(value) for key,value in json.loads((r/'deploy_config.json').read_text())['runtime_overrides'].items()}
mounts.update({pkg+'/v1/ple_offload/in_process.py':candidate/'in_process.py',pkg+'/model_executor/layers/ple_offload_layer.py':v/'vllm/model_executor/layers/ple_offload_layer.py',pkg+'/models/qwen4_exp/nvidia/ple_layer.py':old/'ple_layer.py','/opt/gb10/libple_batch_reader.so':b/('variants/08-async-overlap/libple_batch_reader.so' if args.async_ab else 'variants/06-overlap/libple_batch_reader.so' if args.overlap else 'variants/02-direct-cache/libple_batch_reader.so'),'/opt/gb10/libple_hash_gpu.so':b/'variants/03-gpu-hash/libple_hash_gpu.so','/check.py':candidate/'check_graph.py' if args.overlap else old/'check_in_process.py'})
if args.native_variant:mounts['/opt/gb10/libple_batch_reader.so']=b/'variants'/args.native_variant/'libple_batch_reader.so'
if args.gate_timing:
 mounts['/opt/gb10/libple_hash_gpu.so']=candidate/'libple_hash_gpu.so'
 mounts['/check_hash.py']=b/'variants/03-gpu-hash/check_hash.py'
if args.async_ab:cmd+=['-e','GB10_PLE_ASYNC_AB_STEPS=2']
if args.connector_variant:cmd+=['-e','GB10_PLE_SUBMISSION_POLICY='+args.submission_policy]
if args.telemetry:
 mounts['/opt/gb10/libmapped_wait.so']=candidate/'libmapped_wait.so' if args.gate_timing else b/'variants/07-wait-telemetry/libmapped_wait.so'
 mounts['/check_telemetry.py']=b/'variants/07-wait-telemetry/check_telemetry.py'
if args.overlap:cmd+=['-e','GB10_PLE_OVERLAP=1','-e','QWEN_NSYS_CAPTURE=1']
for dst,src in mounts.items():cmd+=['-v',f'{src}:{dst}:ro']
script='/build/benchmark_staging.py' if args.benchmark else '/check.py'
cmd+=['--entrypoint','/bin/sh','vllm-gb10:v029-roce-qsa55122','-c',f'uv venv --offline --system-site-packages /tmp/ple-check-venv && /tmp/ple-check-venv/bin/python {script}']
if args.telemetry:cmd[-1]+=' && /tmp/ple-check-venv/bin/python /check_telemetry.py'
if args.gate_timing:cmd[-1]+=' && /tmp/ple-check-venv/bin/python /check_hash.py'
result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
log=r/'results/ple-latency-campaign-20260916'/('gate-timing-integration.log' if args.gate_timing else 'async-ab-integration.log' if args.async_ab else 'telemetry-integration.log' if args.telemetry else 'overlap-graph-integration.log' if args.overlap else ('gpu-hash-staging.log' if args.benchmark else 'gpu-hash-integration.log'));log=log.with_name(args.native_variant+'-'+log.name) if args.native_variant else log;log=log.with_name(args.submission_policy+'-'+log.name) if args.connector_variant else log;log.write_text(result.stdout);print(result.stdout);raise SystemExit(result.returncode)
