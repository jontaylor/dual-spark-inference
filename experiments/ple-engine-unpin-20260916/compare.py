"""Descriptive before/after affinity comparison; no causal confidence claim."""
import collections
import csv
import json
from pathlib import Path
import statistics

root = Path('results/ple-engine-unpin-20260916')

def summary(values):
    values = sorted(values)
    i = (len(values) - 1) * .95
    lower = int(i)
    return {'n': len(values), 'median': statistics.median(values),
            'mean': statistics.mean(values),
            'p95': values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (i - lower)}

result = {}
for rank in (0, 1):
    phases = {}
    for phase in ('pinned', 'unpinned'):
        folder = root / phase
        assert json.loads((folder / 'summary.json').read_text())[str(rank)]['errors'] == 0
        assert not (folder / f'rank{rank}.err').read_text().strip()
        with (folder / f'rank{rank}-steps.csv').open() as f:
            rows = list(csv.DictReader(f))
        groups = collections.defaultdict(list)
        for row in rows:
            if row['deferred'] != '1':
                continue
            key = '/'.join(row[k] for k in ('requests', 'rows', 'tokens'))
            groups[key].append(row)
        phases[phase] = groups
    pairs = {}
    for key in phases['pinned'].keys() & phases['unpinned'].keys():
        if min(len(phases[p][key]) for p in phases) < 20:
            continue
        entry = {}
        for metric in ('gpu_hash_to_gate_ms', 'next_period_ms', 'reader_ms', 'gpu_wait_us'):
            entry[metric] = {}
            for phase in phases:
                rows = phases[phase][key]
                if metric == 'next_period_ms':
                    rows = [r for r in rows if r.get('next_deferred') == '1' and
                            all(r.get('next_' + k) == r[k] for k in ('rows', 'requests', 'tokens'))]
                values = [float(r[metric]) for r in rows if r.get(metric)]
                if values:
                    entry[metric][phase] = summary(values)
            if len(entry[metric]) == 2:
                before, after = entry[metric]['pinned'], entry[metric]['unpinned']
                entry[metric]['median_change_pct'] = 100 * (after['median'] / before['median'] - 1) if before['median'] else None
        pairs[key] = entry
    result[rank] = pairs
result['note'] = '60-second pinned and 180-second unpinned sequential windows; fixed policy, same workers, no restart. Exact request/row/token strata; step intervals exclude shape transitions. Context lengths, prefill and workload drift remain uncontrolled. Descriptive evidence only.'
(root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
