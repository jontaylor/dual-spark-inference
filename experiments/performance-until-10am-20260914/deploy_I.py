"""Deploy only at a freshly verified cycle boundary; preserve exact H rollback."""
import json
import shlex
import subprocess
import time
import urllib.request
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[1]
record = {'started': time.time(), 'candidate': 'I-H-plus-MTP5'}
transition = p / 'I-deploy-transition.json'
assert not transition.exists(), 'Existing transition: inspect before retrying'

def save():
    transition.write_text(json.dumps(record, indent=2))

def remote(rank, command, **kwargs):
    if rank:
        command = ['ssh', 'jon@192.168.100.11', shlex.join(command)]
    return subprocess.check_output(command, text=True, timeout=30, **kwargs)

subprocess.run(['python3', str(p / 'preflight_I.py')], check=True)
subprocess.run(['python3', str(p / 'check_I_boundary.py')], check=True)
assert json.loads((p / 'I-boundary-check.json').read_text())['ready'], 'Current workload not drained'
for rank in (0, 1):
    data = remote(rank, ['cat', str(root / 'deploy_config.json')])
    assert json.loads(data) == json.loads((p / f'candidate-H-r{rank}.json').read_text())
    backup = p / f'rollback-H-r{rank}.json'
    assert not backup.exists(), 'Do not overwrite existing exact rollback'
    backup.write_text(data)
with urllib.request.urlopen('http://127.0.0.1:30001/metrics', timeout=5) as response:
    (p / 'H-final-metrics.txt').write_bytes(response.read())
subprocess.run(['python3', str(p / 'summarize_cycle.py'), 'batch-002'], check=True)
record['status'] = 'backups_saved'
save()

# Finish the immutable-file observation BEFORE service cleanup unlinks files.
for rank in (0, 1):
    code = """import json,os,signal,time,sys
from pathlib import Path
p=Path(sys.argv[1]);rank=sys.argv[2]
r=json.loads((p/f'physical-recall-process-r{rank}.json').read_text())
proc=Path(f"/proc/{r['pid']}/stat")
f=proc.read_text().rsplit(') ',1)[1].split()
assert f[19]==r['starttime'] and f[0] not in ('Z','T')
os.kill(r['pid'],signal.SIGTERM)
for _ in range(100):
 d=json.loads((p/f'physical-recall-r{rank}.json').read_text())
 if not d['running']:
  assert not d['errors'],d['errors'];print(json.dumps(d));break
 time.sleep(.1)
else:raise RuntimeError('Physical observer did not finish')
"""
    final = remote(rank, ['python3', '-c', code, str(p / 'physical-H'), str(rank)])
    (p / 'physical-H' / f'final-before-I-r{rank}.json').write_text(final)
subprocess.run(['python3', str(p / 'summarize_physical_recall.py'), str(p / 'physical-H')], check=True)
subprocess.run(['python3', str(p / 'check_I_boundary.py')], check=True)
assert json.loads((p / 'I-boundary-check.json').read_text())['ready'], 'Boundary changed during preparation'
record['status'] = 'observers_finished'
save()

def write_config(rank, data):
    code = "from pathlib import Path;import sys,json;p=Path(sys.argv[1]);d=sys.stdin.read();json.loads(d);t=p.with_suffix('.I-tmp');t.write_text(d);t.replace(p)"
    remote(rank, ['python3', '-c', code, str(root / 'deploy_config.json')], input=data)

stopping = False
try:
    stopping = True
    subprocess.run(['sudo', '-n', 'systemctl', 'stop', 'qwen38-next-qwen-fp8.service'], check=True, timeout=90)
    for rank in (0, 1):
        write_config(rank, (p / f'candidate-I-mtp5-r{rank}.json').read_text())
    record['status'] = 'I_start_requested'
    save()
    subprocess.run(['sudo', '-n', 'systemctl', 'start', '--no-block', 'qwen38-next-qwen-fp8.service'], check=True, timeout=20)
except BaseException as error:
    record['error'] = repr(error)
    if stopping:
        for rank in (0, 1):
            write_config(rank, (p / f'rollback-H-r{rank}.json').read_text())
        subprocess.run(['sudo', '-n', 'systemctl', 'start', '--no-block', 'qwen38-next-qwen-fp8.service'], check=True, timeout=20)
        record['status'] = 'H_restore_start_requested'
    save()
    raise
record['issued_at'] = time.time()
record['remaining'] = ['observe startup, abort known failures', 'verify live config/source hashes',
                       'run I primary and disk gates', 'start new physical observers',
                       'release controller hold and verify resumed']
save()
print(json.dumps(record, indent=2))
