"""Correlate nonblocking CUDA API uprobes with live Nsight GPU metrics.

No inference, server mutation, or GPU operations. Run from the repository root.
The two ranks were captured separately; do not interpret them as paired steps.
"""
import bisect
import json
from pathlib import Path
import sqlite3
import statistics
import sys

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('results/async-scheduling-20260916')


def summarize(values):
    values = sorted(values)
    return dict(n=len(values), mean=statistics.mean(values),
                median=statistics.median(values),
                p95=values[min(len(values)-1, int(.95*len(values)))],
                maximum=max(values)) if values else dict(n=0)


def analyze(rank, minimum_wait_ms=50):
    c = sqlite3.connect(ROOT / f'gpu-metrics-r{rank}.sqlite')
    epoch = c.execute('SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME').fetchone()[0]
    clock = json.loads((ROOT / ('clock.json' if rank == 0 else 'clock-r1.json')).read_text())
    monotonic_start = epoch - clock['wall_minus_monotonic_ns']
    end = c.execute('SELECT max(timestamp) FROM GPU_METRICS').fetchone()[0]
    gaps, event = [], None
    for line in (ROOT / f'host-gap-metrics-r{rank}.txt').read_text().splitlines():
        f = line.split()
        if len(f) < 4:
            continue
        if f[0] == 'E':
            event = (int(f[1]), int(f[3])/1e6)
        elif f[0] == 'K' and event:
            a, duration = event
            b = int(f[1])
            if duration >= minimum_wait_ms and 0 <= a-monotonic_start < b-monotonic_start <= end:
                gaps.append((a-monotonic_start, b-monotonic_start))
            event = None
    starts = [a for a, b in gaps]
    result = dict(rank=rank, session_start_utc_ns=epoch,
                  minimum_wait_ms=minimum_wait_ms,
                  host_gap_ms=summarize([(b-a)/1e6 for a,b in gaps]),
                  host_gap_total_ms=sum(b-a for a,b in gaps)/1e6,
                  metrics={})
    names = dict(c.execute('SELECT metricId,metricName FROM TARGET_INFO_GPU_METRICS'))
    for metric in (6, 7, 8, 9):
        samples = c.execute('SELECT timestamp,value FROM GPU_METRICS WHERE metricId=? ORDER BY timestamp', (metric,)).fetchall()
        within = []
        all_values = []
        longest_zero_ns = zero_start = 0
        was_zero = False
        for t, value in samples:
            if not 0 <= t <= end:
                continue
            all_values.append(value)
            i = bisect.bisect_right(starts, t)-1
            if i >= 0 and t < gaps[i][1]:
                within.append(value)
            if value == 0:
                if not was_zero:
                    zero_start = t
                longest_zero_ns = max(longest_zero_ns, t-zero_start)
            was_zero = value == 0
        result['metrics'][names[metric]] = dict(
            whole_capture=summarize(all_values),
            host_gap=summarize(within),
            whole_capture_zero_fraction=sum(v == 0 for v in all_values)/len(all_values),
            host_gap_zero_samples=sum(v == 0 for v in within),
            longest_zero_sample_span_ms=longest_zero_ns/1e6,
        )
    return result


if __name__ == '__main__':
    result = [analyze(rank, threshold) for rank in (0, 1) for threshold in (10, 50, 100)]
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    for row in result:
        if row['minimum_wait_ms'] == 50:
            print(json.dumps(row, indent=2))
