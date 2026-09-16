import subprocess,time,json,urllib.request,re
from pathlib import Path
p=Path(__file__).parent
while not (p/'ready.json').exists():time.sleep(5)
with (p/'live-correctness.log').open('w') as f:
 subprocess.run(['python3',str(p/'probe_live.py'),'correctness'],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=650)
print('Live correctness passed',flush=True)
start=time.time()
while time.time()-start<600:
 raw=urllib.request.urlopen('http://127.0.0.1:30001/metrics',timeout=10).read().decode()
 n=sum(float(x) for x in re.findall(r'^vllm:kv_offload_store_bytes_total(?:\{[^\n]*\})? ([\d.eE+-]+)$',raw,re.M))
 if n>100000000:
  print('Disk-pressure traffic observed; beginning matched-duration stall capture',flush=True);break
 time.sleep(10)
else:print('No pressure traffic within10min; record below-pressure sample explicitly',flush=True)
subprocess.run(['python3',str(p/'capture.py')],check=True,timeout=220)
print('Post-change observation complete',flush=True)
