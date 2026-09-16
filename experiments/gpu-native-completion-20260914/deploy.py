"""Deploy the verified candidate once, preserving exact prior configurations."""
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

p=Path(__file__).resolve().parent;root=p.parents[1]
for name in ('ownership-v2.log','finish-plan-v2.log','gpu-roundtrip-r0.log','gpu-roundtrip-r1.log'):
    results=[json.loads(line) for line in (p/name).read_text().splitlines() if line.startswith('{')]
    assert results[-1]['passed'],name

def run(rank,args,**kw):
    return subprocess.check_output(['ssh','jon@192.168.100.11',shlex.join(args)] if rank else args,
                                   text=True,timeout=180,**kw)

for rank in (0,1):
    config=json.loads((p/f'candidate-r{rank}.json').read_text())
    assert config['mtp_tokens']==3
    dry=[json.loads(line) for line in (p/f'dry-run-r{rank}.jsonl').read_text().splitlines()]
    args=dry[0]['args'];transfer=json.loads(args[args.index('--kv-transfer-config')+1])
    assert transfer['kv_connector_extra_config']['native_completion_cache'] is True
    for target,source in config['runtime_overrides'].items():
        assert run(rank,['sha256sum',source]).split()[0]==config['runtime_override_sha256'][target],target
    current=json.loads(run(rank,['cat',str(root/'deploy_config.json')]))
    assert current==json.loads((p/f'before-r{rank}.json').read_text()),'Live config changed during preparation'
subprocess.run(['sudo','-n','systemctl','stop','qwen38-next-qwen-fp8.service'],check=True,timeout=180)
for rank in (0,1):
    code="from pathlib import Path;import sys,json;p=Path(sys.argv[1]);s=sys.stdin.read();json.loads(s);t=p.with_suffix('.gpu-native-tmp');t.write_text(s);t.replace(p)"
    run(rank,['python3','-c',code,str(root/'deploy_config.json')],input=(p/f'candidate-r{rank}.json').read_text())
subprocess.run(['sudo','-n','systemctl','start','--no-block','qwen38-next-qwen-fp8.service'],check=True,timeout=20)
(p/'start.json').write_text(json.dumps({'time':time.time(),'mtp':3,
    'policy':'GPU completion events, shared attention, native LRU, allocator-triggered disk writes'},indent=2)+'\n')
print('GPU native completion candidate restart requested',flush=True)
