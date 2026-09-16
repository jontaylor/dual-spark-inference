"""Attribute sampled zero-SM time to host CUDA waits, not causal GPU idle."""
import bisect
import json
from pathlib import Path
import sqlite3
import sys

root = Path(sys.argv[1])
result = []
for rank in (0, 1):
    db = sqlite3.connect(root / f'gpu-metrics-r{rank}.sqlite')
    epoch = db.execute('SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME').fetchone()[0]
    clock = json.loads((root / ('clock.json' if rank == 0 else 'clock-r1.json')).read_text())
    offset = clock['wall_minus_monotonic_ns'] - epoch
    samples = db.execute('SELECT timestamp,value FROM GPU_METRICS WHERE metricId=7 AND timestamp>=0 ORDER BY timestamp').fetchall()
    timestamps = [t for t,v in samples]
    zero_prefix = [0]
    for t,v in samples:
        zero_prefix.append(zero_prefix[-1] + (v == 0))
    grouped = {}
    for line in (root / f'host-gap-metrics-r{rank}.txt').read_text().splitlines():
        fields = line.split()
        if len(fields) != 6 or fields[0] != 'W':
            continue
        _, begin, end, tid, name, status = fields
        begin, end = int(begin)+offset, int(end)+offset
        begin, end = max(0, begin), min(timestamps[-1], end)
        if end <= begin:
            continue
        key = (int(tid), name)
        entry = grouped.setdefault(key, {'calls':0, 'host_ms':0, 'zero_sm_samples':0})
        entry['calls'] += 1
        entry['host_ms'] += (end-begin)/1e6
        lo, hi = bisect.bisect_left(timestamps, begin), bisect.bisect_left(timestamps, end)
        entry['zero_sm_samples'] += zero_prefix[hi]-zero_prefix[lo]
    result.append({'rank':rank, 'whole_zero_sm_samples':zero_prefix[-1],
                   'waits':[dict(tid=tid,api=name,**v) for (tid,name),v in grouped.items()]})
output = {'limits':'Only calls lasting >=1 ms; truncated to metric capture. At 10 kHz each zero sample represents approximately 0.1 ms. API waits can overlap across threads or nested calls: do not sum categories. CUDA waits often include useful GPU work, and zero-SM does not exclude copies.', 'ranks':result}
(root / 'wait-summary.json').write_text(json.dumps(output, indent=2)+'\n')
print(json.dumps(output, indent=2))
