"""Release existing controller hold only after inspected J validation gates."""
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[1]
expected = '7dadca0c648ba60cf978654d2f6a4a19f40b5ee6a7ad456985284586272ab845'
assert hashlib.sha256((root / 'deploy_config.json').read_bytes()).hexdigest() == expected
primary = json.loads((p / 'validation-J/summary.json').read_text())
disk = json.loads((p / 'J-disk-validation.json').read_text())
assert primary['passed'] and primary['config_sha256'] == expected and disk['passed']
runtime=json.loads((p/'J-runtime-verified.json').read_text())
assert runtime['passed'] and runtime['head_config_sha256']==expected
assert all(r['verified_overrides']==18 for r in runtime['ranks'])
assert json.loads((p/'growth-validation-J/summary.json').read_text())['passed']
oracle = json.loads((p / 'disk-oracle-J/summary.json').read_text())
assert oracle['disk_load_bytes'] > 0 and oracle['independent_load_bytes'] == 0
assert oracle['tokens_equal'] and oracle['scores_equal'] and oracle['exact_prefix_matches']
hold = json.loads((p / 'reservation-next-boundary-status.json').read_text())
fields = Path(f"/proc/{hold['watchdog_pid']}/stat").read_text().rsplit(') ', 1)[1].split()
assert fields[19] == hold['watchdog_starttime'] and fields[0] not in ('Z', 'T')
assert time.time() < hold['deadline'] and hold['status'] == 'boundary_ready'
with urllib.request.urlopen('http://127.0.0.1:30001/health', timeout=5) as response:
    assert response.status == 200
for rank in (0, 1):
    path = p / 'physical-J' / f'physical-recall-r{rank}.json'
    raw = path.read_text() if rank == 0 else subprocess.check_output(['ssh', 'jon@192.168.100.11', 'cat', str(path)], text=True, timeout=20)
    observer = json.loads(raw)
    assert observer['running'] and not observer['errors'] and time.time() - observer['time'] < 90
code = "from pathlib import Path;f=Path('/proc/1178958/stat').read_text().rsplit(') ',1)[1].split();assert f[19]=='9802967' and f[0]=='T';print('held identity verified')"
import shlex
subprocess.run(['ssh', 'jon@192.168.0.167', shlex.join(['python3', '-c', code])], check=True, timeout=20)
with urllib.request.urlopen('http://127.0.0.1:30001/metrics', timeout=5) as response:
    (p / 'J-pre-resume-metrics.txt').write_bytes(response.read())
(p / 'reservation-next-boundary-release').touch(exist_ok=False)
(p / 'J-release-requested.json').write_text(json.dumps({'time': time.time(), 'config_sha256': expected, 'remaining': 'Verify hold watchdog reports resumed and actual controller state/new cycle launch.'}, indent=2))
print('Release requested through existing watchdog; actual resumption still to verify.')
