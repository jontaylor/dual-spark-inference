"""Release existing controller hold only after inspected I2 validation gates."""
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[1]
expected = '3a57ca682283e2e79da8d00b722ce39cca9027e6ab85f4185caf46f3b53f40b9'
assert hashlib.sha256((root / 'deploy_config.json').read_bytes()).hexdigest() == expected
primary = json.loads((p / 'validation-I2/summary.json').read_text())
disk = json.loads((p / 'I2-disk-validation.json').read_text())
assert primary['passed'] and primary['config_sha256'] == expected and disk['passed']
oracle = json.loads((p / 'disk-oracle-I2/summary.json').read_text())
assert oracle['disk_load_bytes'] > 0 and oracle['independent_load_bytes'] == 0
assert oracle['tokens_equal'] and oracle['scores_equal'] and oracle['exact_prefix_matches']
hold = json.loads((p / 'mtp5-boundary-status.json').read_text())
fields = Path(f"/proc/{hold['watchdog_pid']}/stat").read_text().rsplit(') ', 1)[1].split()
assert fields[19] == hold['watchdog_starttime'] and fields[0] not in ('Z', 'T')
assert time.time() < hold['deadline'] and hold['status'] == 'boundary_ready'
with urllib.request.urlopen('http://127.0.0.1:30001/health', timeout=5) as response:
    assert response.status == 200
for rank in (0, 1):
    path = p / 'physical-I2' / f'physical-recall-r{rank}.json'
    raw = path.read_text() if rank == 0 else subprocess.check_output(['ssh', 'jon@192.168.100.11', 'cat', str(path)], text=True, timeout=20)
    observer = json.loads(raw)
    assert observer['running'] and not observer['errors'] and time.time() - observer['time'] < 90
code = "from pathlib import Path;f=Path('/proc/1178958/stat').read_text().rsplit(') ',1)[1].split();assert f[19]=='9802967' and f[0]=='T';print('held identity verified')"
import shlex
subprocess.run(['ssh', 'jon@192.168.0.167', shlex.join(['python3', '-c', code])], check=True, timeout=20)
with urllib.request.urlopen('http://127.0.0.1:30001/metrics', timeout=5) as response:
    (p / 'I2-pre-resume-metrics.txt').write_bytes(response.read())
(p / 'mtp5-boundary-release').touch(exist_ok=False)
(p / 'I2-release-requested.json').write_text(json.dumps({'time': time.time(), 'config_sha256': expected, 'remaining': 'Verify hold watchdog reports resumed and actual controller state/new cycle launch.'}, indent=2))
print('Release requested through existing watchdog; actual resumption still to verify.')
