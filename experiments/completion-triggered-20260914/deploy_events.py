import json,subprocess,shlex,time,hashlib,ast
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1]
def run(rank,args,**kw):return subprocess.check_output((['ssh','jon@192.168.100.11',shlex.join(args)] if rank else args),text=True,timeout=180,**kw)
overrides={'distributed/kv_transfer/kv_connector/v1/gb10_completion.py':'completion.events.py','distributed/kv_transfer/kv_connector/v1/gb10_aligned_offloading_connector.py':'connector.events.py','v1/core/sched/scheduler.py':'scheduler.events.py'}
for name in overrides.values():ast.parse((p/name).read_text())
assert json.loads((p/'cpu-gates.log').read_text().splitlines()[-1])['passed']
run(1,['mkdir','-p',str(p)])
for name in overrides.values():subprocess.run(['scp','-q',str(p/name),'jon@192.168.100.11:'+str(p/name)],check=True)
for rank in [0,1]:
 cfg=json.loads((p/f'mtp3-baseline-r{rank}.json').read_text())
 for target,name in overrides.items():
  cfg['runtime_overrides'][target]=str(p/name)
  cfg['runtime_override_sha256'][target]=hashlib.sha256((p/name).read_bytes()).hexdigest()
 (p/f'candidate-events-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
subprocess.run(['sudo','-n','systemctl','stop','qwen38-next-qwen-fp8.service'],check=True,timeout=180)
for rank in [0,1]:
 code="from pathlib import Path;import sys,json;p=Path(sys.argv[1]);s=sys.stdin.read();json.loads(s);t=p.with_suffix('.events-tmp');t.write_text(s);t.replace(p)"
 run(rank,['python3','-c',code,str(root/'deploy_config.json')],input=(p/f'candidate-events-r{rank}.json').read_text())
subprocess.run(['sudo','-n','systemctl','start','--no-block','qwen38-next-qwen-fp8.service'],check=True,timeout=20)
(p/'events-start.json').write_text(json.dumps({'time':time.time(),'status':'candidate start requested','mtp':3,'checkpoint_policy':'completion and suspend events only'},indent=2))
print('Completion-event candidate start requested',flush=True)
