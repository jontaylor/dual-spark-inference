"""Explicit recovery entrypoint; never executed merely because loading is slow."""
import json,subprocess,shlex
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1]
subprocess.run(['sudo','-n','systemctl','stop','qwen38-next-qwen-fp8.service'],check=True,timeout=180)
for rank in [0,1]:
 body=(p/f'before-r{rank}.json').read_text();json.loads(body)
 args=['python3','-c','from pathlib import Path;import sys;p=Path(sys.argv[1]);s=sys.stdin.read();q=p.with_suffix(".rollback-tmp");q.write_text(s);q.replace(p)',str(root/'deploy_config.json')]
 subprocess.run(['ssh','jon@192.168.100.11',shlex.join(args)] if rank else args,input=body,text=True,check=True,timeout=60)
subprocess.run(['sudo','-n','systemctl','start','--no-block','qwen38-next-qwen-fp8.service'],check=True)
