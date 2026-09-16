import time,subprocess,json,urllib.request
from pathlib import Path
p=Path(__file__).parent
start=time.time()
while not (p/'ready.json').exists():
 if time.time()-start>900:raise TimeoutError('Readiness not achieved')
 time.sleep(5)
with (p/'live-correctness.log').open('w') as f:
 subprocess.run(['python3',str(p/'probe_live.py'),'correctness'],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=650)
print('Live correctness passed',flush=True)
