"""Read-only live process and backend gate for the MTP5 transition."""
import json
import subprocess
import time
import urllib.request
from pathlib import Path

p = Path(__file__).resolve().parent
hold = json.loads((p / 'mtp5-boundary-status.json').read_text())
code = r'''
import json,sys
from pathlib import Path
h=json.load(sys.stdin)
def state(pid, start):
 try:f=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
 except FileNotFoundError:return 'gone'
 return f[0] if f[19]==start else 'identity_changed'
root=Path(h['current']);campaign=json.loads((root/'campaign.json').read_text())
owners=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  args=(proc/'cmdline').read_bytes().split(b'\0')
  fields=(proc/'stat').read_text().rsplit(') ',1)[1].split()
 except (FileNotFoundError,PermissionError,ProcessLookupError):continue
 if str(root/'run.py').encode() in args:
  owners.append({'pid':int(proc.name),'state':fields[0],'starttime':fields[19]})
print(json.dumps({'controller_state':state(h['pid'],h['starttime']),
 'supervisor_state':state(h['supervisor'],h['supervisor_starttime']),
 'campaign_status':campaign['status'],'owners':owners}))
'''
command = ['ssh', 'jon@192.168.0.167', 'python3', '-c', __import__('shlex').quote(code)]
remote = json.loads(subprocess.check_output(command, input=json.dumps(hold), text=True, timeout=20))
with urllib.request.urlopen('http://127.0.0.1:30001/metrics', timeout=5) as response:
    raw = response.read().decode()
gauges = [line for line in raw.splitlines() if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
idle = len(gauges) >= 2 and all(float(line.rsplit(' ', 1)[1]) == 0 for line in gauges)
watch = Path(f"/proc/{hold['watchdog_pid']}/stat")
fields = watch.read_text().rsplit(') ', 1)[1].split() if watch.exists() else []
watch_alive = bool(fields and fields[19] == hold['watchdog_starttime'] and fields[0] not in ('Z', 'T'))
ready = (watch_alive and time.time() < hold['deadline'] - 1200
         and remote['controller_state'] == 'T'
         and remote['supervisor_state'] in ('gone', 'Z')
         and remote['campaign_status'] == 'finished'
         and not remote['owners'] and idle)
result = {'time': time.time(), 'ready': ready, 'remote': remote,
          'backend_idle': idle, 'gauges': gauges, 'hold_watchdog_alive': watch_alive,
          'scope': 'Observation only; rerun immediately before transition. No process is signalled.'}
(p / 'I-boundary-check.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
