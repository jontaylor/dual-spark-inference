"""Run the existing state-copy and worker lifetime scenarios on a real GPU.

Uses tiny isolated buffers, never attaches to or modifies the serving worker.
The runpy-like transformation keeps the same assertions as the CPU scenarios.
"""
import json
import os
from pathlib import Path
import sys
import time
assert os.environ.get('TRITON_INTERPRET') != '1'
import torch
root=Path(sys.argv[1]);candidate=Path(sys.argv[2])
start=time.time()
print(json.dumps(dict(event='device_check_start',wall=start)),flush=True)
torch.set_default_device('cuda')
for name in ('check_terminal_copy.py','check_terminal_worker.py'):
 code=(root/name).read_text()
 code=code.replace("assert os.environ.get('TRITON_INTERPRET') == '1'", "assert os.environ.get('TRITON_INTERPRET') != '1'")
 code=code.replace("torch.device('cpu')", "torch.device('cuda')")
 code=code.replace('.numpy().tobytes()', '.cpu().numpy().tobytes()')
 code=code.replace("mode='Triton CPU interpreter'", "mode='CUDA device'")
 sys.argv=[name,str(candidate/'terminal_state.py') if name=='check_terminal_copy.py' else str(candidate)]
 exec(compile(code,str(root/name),'exec'),{'__name__':'__main__','__file__':str(root/name)})
 torch.cuda.synchronize()
print(json.dumps(dict(event='device_check_complete',passed=True,wall=time.time(),
 elapsed_seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated(),
 peak_reserved=torch.cuda.max_memory_reserved())),flush=True)
