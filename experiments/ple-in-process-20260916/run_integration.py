from pathlib import Path
import json
import subprocess
v=Path('/home/jon/vllm-gb10-kv-paging');b=Path(__file__).parent;pkg='/usr/local/lib/python3.12/dist-packages/vllm'
cmd=['docker','run','--rm','--gpus','all','--network','none','--ipc','host','--ulimit','memlock=-1','--security-opt',f'seccomp={b}/seccomp-uring.json','-e','GB10_PLE_IN_PROCESS=1','-e','GB10_PLE_MAPPED_TRANSPORT=1','-e','VLLM_PLE_CPU_OFFLOAD=1','-e','VLLM_PLE_LOCAL_TP=1','-e','GB10_PLE_ROW_CACHE_MB=1','-e','VLLM_PLE_PACKED_TABLE_DIR=/tmp/ple-test-table']
mounts={pkg+'/'+k:Path(s) for k,s in json.load(open(b.parents[1]/'deploy_config.json'))['runtime_overrides'].items()}
mounts.update({pkg+'/v1/ple_offload/in_process.py':v/'vllm/v1/ple_offload/in_process.py',pkg+'/model_executor/layers/ple_offload_layer.py':v/'vllm/model_executor/layers/ple_offload_layer.py',pkg+'/models/qwen4_exp/nvidia/ple_layer.py':b/'ple_layer.py','/opt/gb10/libple_batch_reader.so':b/'libple_batch_reader.so','/check.py':b/'check_in_process.py'})
for dst,src in mounts.items():cmd+=['-v',f'{src}:{dst}:ro']
cmd+=['--entrypoint','/bin/sh','vllm-gb10:v029-roce-qsa55122','-c','uv venv --offline --system-site-packages /tmp/ple-check-venv && /tmp/ple-check-venv/bin/python /check.py']
result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(b/'integration.log').write_text(result.stdout);print(result.stdout);raise SystemExit(result.returncode)
