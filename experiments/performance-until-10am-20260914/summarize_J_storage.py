"""Observe J physical blob lifetimes and worker/device I/O without model probes."""
import json
import subprocess
import time
from pathlib import Path

p = Path(__file__).resolve().parent
subprocess.run(['python3', str(p / 'summarize_physical_recall.py'),
                str(p / 'physical-J')], check=True, stdout=subprocess.DEVNULL)
physical = json.loads((p / 'physical-J/physical-recall-summary.json').read_text())
written = sum(r['cohorts'][name]['bytes'] for r in physical['ranks']
              for name in ['new_live', 'new_retired'])
unread = sum(r['cohorts']['new_retired']['unread_bytes'] for r in physical['ranks'])
lifetime = {'time': time.time(), 'new_physical_bytes_observed': written,
            'retired_unread_bytes': unread,
            'confirmed_unread_lower_bound': unread / written if written else None,
            'observer_ages_seconds': [r['age_seconds'] for r in physical['ranks']],
            'scope': 'New physical generations only, before any server cleanup. Live unread bytes excluded from confirmed waste. Metadata identifies any reader, not its PID. Cohort is still developing.'}
(p / 'J-unread-write-lower-bound.json').write_text(json.dumps(lifetime, indent=2))

start = json.loads((p / 'J-release-requested.json').read_text())['time']
rows = [json.loads(line) for line in (p / 'storage-io.jsonl').read_text().splitlines()]
rows = [r for r in rows if r['time'] >= start and len(r.get('ranks', [])) == 2
        and all('workers' in rank for rank in r['ranks'])]
assert len(rows) >= 2
a, b = rows[0], rows[-1]
reports = []
for rank in (0, 1):
    x = next(r for r in a['ranks'] if r['rank'] == rank)
    y = next(r for r in b['ranks'] if r['rank'] == rank)
    assert len(x['workers']) == len(y['workers']) == 1
    wx, wy = x['workers'][0], y['workers'][0]
    assert (wx['pid'], wx['starttime']) == (wy['pid'], wy['starttime'])
    io = {k: wy['io'][k] - wx['io'][k]
          for k in ['read_bytes', 'write_bytes', 'cancelled_write_bytes']}
    assert all(value >= 0 for value in io.values())
    reports.append({'rank': rank, 'worker_pid': wx['pid'],
                    'worker_starttime': wx['starttime'], 'worker_io_delta': io,
                    'device_read_bytes': y['device_read_bytes'] - x['device_read_bytes'],
                    'device_write_bytes': y['device_write_bytes'] - x['device_write_bytes']})
result = {'time': time.time(), 'start': a['time'], 'end': b['time'],
          'seconds': b['time'] - a['time'], 'start_offset_seconds': a['time'] - start,
          'ranks': reports,
          'scope': 'Same worker PID/starttime throughout interval endpoints. Worker I/O includes all worker files; device I/O includes all host activity. No exclusive attribution of device writes to KV payload.'}
(p / 'J-workload-storage-io.json').write_text(json.dumps(result, indent=2))
print(json.dumps({'lifetimes': lifetime, 'io': result}, indent=2))
