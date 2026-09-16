"""Verify J code in both running containers, not merely staged source files."""
import hashlib,json,subprocess,shlex,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1];rows=[]
for rank in (0,1):
 expected=json.loads((p/f'candidate-J-reservation-r{rank}.json').read_text())
 cmd=['cat',str(root/'deploy_config.json')];cmd=cmd if rank==0 else ['ssh','jon@192.168.100.11',shlex.join(cmd)]
 assert json.loads(subprocess.check_output(cmd,text=True,timeout=20))==expected
 code="""import hashlib,json,sys;from pathlib import Path
d=json.load(sys.stdin);root=Path('/usr/local/lib/python3.12/dist-packages/vllm')
for k,v in d['runtime_override_sha256'].items():assert hashlib.sha256((root/k).read_bytes()).hexdigest()==v,k
print(json.dumps({'verified_overrides':len(d['runtime_override_sha256'])}))
"""
 cmd=['docker','exec','-i',f'qwen38-kv-paging-r{rank}','python3','-c',code];cmd=cmd if rank==0 else ['ssh','jon@192.168.100.11',shlex.join(cmd)]
 r=json.loads(subprocess.check_output(cmd,input=json.dumps(expected),text=True,timeout=30))
 cmd=['docker','inspect',f'qwen38-kv-paging-r{rank}','--format','{{json .State}}'];cmd=cmd if rank==0 else ['ssh','jon@192.168.100.11',shlex.join(cmd)]
 state=json.loads(subprocess.check_output(cmd,text=True,timeout=20));assert state['Running'] and not state['OOMKilled'];rows.append({'rank':rank,**r,'state':state})
s={'time':time.time(),'passed':True,'ranks':rows,'head_config_sha256':hashlib.sha256((root/'deploy_config.json').read_bytes()).hexdigest(),'scope':'Both running containers source hashes and state; model gates are separate.'};(p/'J-runtime-verified.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
