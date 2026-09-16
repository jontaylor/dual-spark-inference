import json,subprocess,shlex,time,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1]
def run(rank,args,**kw):
 return subprocess.check_output((['ssh','jon@192.168.100.11',shlex.join(args)] if rank else args),text=True,timeout=90,**kw)
for rank in [0,1]:
 raw=run(rank,['cat',str(root/'deploy_config.json')]);cfg=json.loads(raw)
 backup=p/f'before-r{rank}.json';assert not backup.exists();backup.write_text(raw)
 assert cfg['mtp_tokens']==5
 cfg['mtp_tokens']=3
 # Preserve page geometry, graph coverage and all source overrides for baseline.
 (p/f'mtp3-baseline-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
subprocess.run(['sudo','-n','systemctl','stop','qwen38-next-qwen-fp8.service'],check=True,timeout=90)
for rank in [0,1]:
 code="from pathlib import Path;import sys,json;p=Path(sys.argv[1]);s=sys.stdin.read();json.loads(s);t=p.with_suffix('.mtp3-tmp');t.write_text(s);t.replace(p)"
 run(rank,['python3','-c',code,str(root/'deploy_config.json')],input=(p/f'mtp3-baseline-r{rank}.json').read_text())
subprocess.run(['sudo','-n','systemctl','start','--no-block','qwen38-next-qwen-fp8.service'],check=True,timeout=20)
(p/'mtp3-start.json').write_text(json.dumps({'time':time.time(),'status':'start requested','changed':['mtp_tokens:5->3']},indent=2))
print('MTP3 start requested; cache geometry and all source overrides unchanged',flush=True)
