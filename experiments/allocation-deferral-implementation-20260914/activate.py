"""Explicit live-test activation; never called by preparation or unit tests."""
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
ROOT=P.parents[1]
def call(rank,args,**kwargs):
    return subprocess.check_output(['ssh','jon@192.168.100.11',shlex.join(args)] if rank else args,text=True,timeout=90,**kwargs)
configs={}
for rank in (0,1):
    before=json.loads((P/f'before-r{rank}.json').read_text())
    assert json.loads(call(rank,['cat',str(ROOT/'deploy_config.json')]))==before,'Serving config changed since preparation'
    cfg=configs[rank]=json.loads((P/f'candidate-r{rank}.json').read_text())
    for dest,src in cfg['runtime_overrides'].items():
        assert call(rank,['sha256sum',src]).split()[0]==cfg['runtime_override_sha256'][dest],(rank,dest)
checks=json.loads((P/'checks.json').read_text())
assert checks and all(r['passed'] for r in checks.values())
message='Starting the prepared allocation-aware deferral live test: source blocks reserved until both-rank copy ACK, affected requests defer; no native eviction batch fence. MTP3, numerical kernels, cache retention and eight CPU staging slots unchanged. Please preserve representative workload sampling/cadence/retries across one restart.'
subprocess.run(['ssh','jon@192.168.0.167',shlex.join(['codex','queue','--thread','01a08cc2-ed05-7073-aa5a-7bebae318c0e','--message',message])],check=True,timeout=30)
subprocess.run(['sudo','-n','systemctl','stop','qwen38-next-qwen-fp8.service'],check=True,timeout=180)
try:
    for rank in (0,1):
        code='from pathlib import Path;import sys,json;p=Path(sys.argv[1]);s=sys.stdin.read();json.loads(s);q=p.with_suffix(".deferral-tmp");q.write_text(s);q.replace(p)'
        call(rank,['python3','-c',code,str(ROOT/'deploy_config.json')],input=json.dumps(configs[rank],indent=2))
    subprocess.run(['sudo','-n','systemctl','start','--no-block','qwen38-next-qwen-fp8.service'],check=True,timeout=30)
except BaseException:
    subprocess.run(['python3',str(P/'rollback.py')],check=True)
    raise
print('Candidate restart requested; readiness and live correctness still need checking')
