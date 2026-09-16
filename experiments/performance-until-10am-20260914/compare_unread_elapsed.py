"""Conservative unread-retirement bounds within matched elapsed workload windows."""
import json
import subprocess
import time
from pathlib import Path

p = Path(__file__).resolve().parent
comparison = json.loads((p / 'H-I3-J-matched-elapsed.json').read_text())
reports = []
for row in comparison['rows']:
    label = row['config']
    counts = []
    for rank in (0, 1):
        path = p / f'physical-{label}' / f'physical-recall-r{rank}.jsonl'
        raw = path.read_text() if rank == 0 else subprocess.check_output(
            ['ssh', 'jon@192.168.100.11', 'cat', str(path)], text=True, timeout=20)
        events = [json.loads(line) for line in raw.splitlines() if line]
        eligible = [e for e in events if not e['preexisting']
                    and row['start'] <= e['attached'] <= e['time'] <= row['end']]
        counts.append({'rank': rank,
                       'retired_generations': len(eligible),
                       'retired_unread_bytes': sum(e['size'] for e in eligible
                                                   if not e['observed_read'])})
    unread = sum(r['retired_unread_bytes'] for r in counts)
    assert 0 <= unread <= row['payload_written']
    reports.append({'config': label, 'start': row['start'], 'end': row['end'],
                    'seconds': row['seconds'], 'written_payload_bytes': row['payload_written'],
                    'retired_unread_bytes': unread,
                    'confirmed_unread_lower_bound': unread / row['payload_written']
                    if row['payload_written'] else None, 'ranks': counts})
result = {'time': time.time(), 'rows': reports,
          'scope': 'Only new blobs attached and retired inside each matched metric window. Live unread blobs and preexisting blobs excluded. Attachment/counter sampling offsets can conservatively exclude boundary blobs. Metadata identifies any reader. Different trajectories/concurrency; this is not a demonstrated policy saving.'}
(p / 'H-I3-J-unread-matched-elapsed.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
