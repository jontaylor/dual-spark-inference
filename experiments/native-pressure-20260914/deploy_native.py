import ast
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[1]
assert json.loads((p/'cpu-gates.log').read_text().splitlines()[-1])['passed']

def run(rank, args, **kw):
    return subprocess.check_output(
        ['ssh','jon@192.168.100.11',shlex.join(args)] if rank else args,
        text=True, timeout=180, **kw)

run(1,['mkdir','-p',str(p)])
for name in ('native_pressure.py','block_pool.native.py','connector.native.py','completion.native.py'):
    ast.parse((p/name).read_text())
    subprocess.run(['scp','-q',str(p/name),'jon@192.168.100.11:'+str(p/name)],check=True)
for rank in (0,1):
    cfg=json.loads((p/f'candidate-r{rank}.json').read_text())
    for target,source in cfg['runtime_overrides'].items():
        digest=run(rank,['sha256sum',source]).split()[0]
        assert digest == cfg['runtime_override_sha256'][target], (rank,target)
subprocess.run(['sudo','-n','systemctl','stop','qwen38-next-qwen-fp8.service'],check=True,timeout=180)
for rank in (0,1):
    code="from pathlib import Path;import sys,json;p=Path(sys.argv[1]);s=sys.stdin.read();json.loads(s);t=p.with_suffix('.native-tmp');t.write_text(s);t.replace(p)"
    run(rank,['python3','-c',code,str(root/'deploy_config.json')],input=(p/f'candidate-r{rank}.json').read_text())
subprocess.run(['sudo','-n','systemctl','start','--no-block','qwen38-next-qwen-fp8.service'],check=True,timeout=20)
(p/'start.json').write_text(json.dumps({'time':time.time(),'mtp':3,'policy':'native allocation eviction + active pressure parking'},indent=2))
print('Native pressure candidate start requested',flush=True)
