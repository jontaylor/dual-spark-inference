"""Summarize sampled admission pressure; never infer readiness from queue length."""
import datetime as dt
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/async-scheduling-20260916'
rows = []
for line in (OUT / 'campaign-backend-metrics.jsonl').read_text().splitlines():
    record = json.loads(line)
    timestamp = dt.datetime.fromisoformat(record['time']).timestamp()
    values = {}
    for sample in record.get('samples', []):
        match = re.fullmatch(r'vllm:(\w+)\{(.*)\} ([\d.eE+\-]+)', sample)
        if not match:
            continue
        name, labels, value = match.groups()
        if name == 'num_requests_waiting_by_reason':
            reason = re.search(r'reason="([^"]+)"', labels)
            name += '_' + reason.group(1)
        values[name] = float(value)
    if 'num_requests_running' in values and 'num_requests_waiting' in values:
        rows.append((timestamp, values))

def summarize(start, end):
    selected = [(t,v) for t,v in rows if start <= t < end]
    groups = {}
    for lower, upper in [(0, 0), (1, 8), (9, 16), (17, 24), (25, 32)]:
        group = [v for _,v in selected if lower <= v['num_requests_running'] <= upper]
        if not group:
            continue
        groups[f'{lower}-{upper}'] = {
            'samples': len(group),
            'with_waiters': sum(v['num_requests_waiting'] > 0 for v in group),
            'with_capacity_waiters': sum(v.get('num_requests_waiting_by_reason_capacity', 0) > 0 for v in group),
            'with_deferred_waiters': sum(v.get('num_requests_waiting_by_reason_deferred', 0) > 0 for v in group),
            'mean_waiters': sum(v['num_requests_waiting'] for v in group)/len(group),
        }
    return {'samples': len(selected), 'by_running_requests': groups}

def utc(text):
    return dt.datetime.fromisoformat(text + '+00:00').timestamp()

result = {
    'source': 'campaign-backend-metrics.jsonl',
    'limits': 'Approximately 5-second samples, not per-step admission tracing. Queue labels do not prove readiness or identify sync overhead. Post-recovery interval includes separate measurement interference; no throughput comparison is made.',
    'post_recovery': summarize(utc('2026-09-16T17:48:08'), float('inf')),
    'low_concurrency_capture': summarize(utc('2026-09-16T18:34:00'), utc('2026-09-16T18:34:31')),
}
(OUT / 'admission-summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
