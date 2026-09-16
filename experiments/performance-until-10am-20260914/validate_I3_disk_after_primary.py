import json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;start=time.monotonic();py=str(p.parents[1]/'.venv/bin/python')
while time.monotonic()-start<900:
 f=p/'validation-I3/summary.json'
 if f.exists():
  try:s=json.loads(f.read_text())
  except json.JSONDecodeError:time.sleep(.5);continue
  if 'passed' in s:
   assert s['passed'],'Primary model gates failed; no disk pressure submitted';break
 time.sleep(2)
else:raise TimeoutError('Primary gate wait expired; inspect live validator')
print('Primary gates passed; starting controlled disk coverage',flush=True)
with (p/'disk-probe-I3.log').open('w') as log:r=subprocess.run([py,str(p/'disk_probe_I3.py')],stdout=log,stderr=subprocess.STDOUT)
s=json.loads((p/'disk-validation-I3/summary.json').read_text())
assert s['tokens_equal'] and s['scores_equal'] and s['restored_cached']==s['expected_cached']
assert s['after']['num_preemptions_total']==s['before']['num_preemptions_total']
if r.returncode:assert s['disk_load_bytes']==0,'Unexpected initial disk-probe failure'
print('Initial target diskload',s['disk_load_bytes'],'; checking independent disk-versus-resident oracle',flush=True)
with (p/'disk-oracle-I3.log').open('w') as log:subprocess.run([py,str(p/'check_existing_disk_I3.py')],stdout=log,stderr=subprocess.STDOUT,check=True)
(p/'I3-disk-validation.json').write_text(json.dumps({'initial_probe_exit':r.returncode,'initial_target_disk_bytes':s['disk_load_bytes'],'independent_oracle':str(p/'disk-oracle-I3/summary.json'),'passed':True},indent=2))
print('Independent disk oracle passed; inspect then resume recorded workload.',flush=True)
