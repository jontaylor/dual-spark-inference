from pathlib import Path
import sys,json,subprocess,hashlib
r=Path('/home/jon/dual-spark-inference-kv-paging');e=r/'experiments/patch-attribution-20260913';rank=int(sys.argv[1])
c=json.loads((e/'config.before.json').read_text());args=json.loads((e/'head-args.json').read_text())
for flag in ['--kv-transfer-config','--scheduler-cls']:
 i=args.index(flag);del args[i:i+2]
args[args.index('--node-rank')+1]=str(rank)
if rank:
 for flag in ['--host','--port']:
  i=args.index(flag);del args[i:i+2]
 args.remove('--trust-request-chat-template');args.append('--headless')
name='qwen38-kv-paging-r'+str(rank)
x=subprocess.run(['docker','inspect','--format','{{.State.Running}}',name],text=True,capture_output=True)
if x.returncode==0:
 assert x.stdout.strip()=='false','Refusing to replace running container'
 subprocess.run(['docker','rm',name],check=True)
base='vllm/vllm-openai@sha256:18372a7224938643461b846fb64c5c9d3d6e9727e82caf2dc3043e620c9d4d7a'
cmd=['docker','run','-d','--name',name,'--gpus','all','--network','host','--ipc','host','--cap-add','SYS_NICE','--ulimit','memlock=-1','--ulimit','stack=67108864','--device','/dev/infiniband:/dev/infiniband','--entrypoint','python3']
mounts={
 str(Path(c['paths']['hf_cache'])/('models--'+c['model_id'].replace('/','--'))):'/model-store',
 str(r/'container_entry.py'):'/opt/container_entry.py',
 str(e/'ple_layer.compat.py'):'/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py',
}
if rank==0:mounts[c['api_key_file']]='/run/secrets/inference-api-key'
for src,dst in mounts.items():cmd+=['-v',src+':'+dst+':ro']
cmd+=['-v',c['paths']['runtime_cache']+':/root/.cache/vllm']
env={'GLOO_SOCKET_IFNAME':c['interface'],'NCCL_SOCKET_IFNAME':c['interface'],'TP_SOCKET_IFNAME':c['interface'],'NCCL_IB_DISABLE':'0','NCCL_IB_HCA':c['ib_hca'],'NCCL_IB_GID_INDEX':'3','NCCL_IB_AUTO_DETECT':'0','NCCL_DEBUG':'WARN','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','VLLM_USE_V2_MODEL_RUNNER':'1','VLLM_HOST_IP':c['head_ip'] if rank==0 else c['worker_ip'],'VLLM_ENGINE_READY_TIMEOUT_S':'3600','VLLM_ALLOW_LONG_MAX_MODEL_LEN':'1'}
for k,v in env.items():cmd+=['-e',k+'='+v]
cmd+=[base,'/opt/container_entry.py',*args]
(e/('upstream-command-r'+str(rank)+'.json')).write_text(json.dumps(cmd,indent=2))
subprocess.run(cmd,check=True)
