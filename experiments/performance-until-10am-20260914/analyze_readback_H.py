"""Join paired windows to measured concurrency; never infer causal speedup."""
import json
from pathlib import Path

p = Path(__file__).resolve().parent
out = p / 'readback-paired-H'
experiment = json.loads((out / 'summary.json').read_text())
samples = [json.loads(line) for line in (p / 'metrics.jsonl').read_text().splitlines()]
samples = [row for row in samples if row.get('metrics')]

def total(metrics, name):
    return sum(v for k, v in metrics.items() if k.startswith('vllm:' + name + '{'))

rows = []
for phase in experiment['phases']:
    raw = json.loads((out / f"metrics-{phase['index']}.json").read_text())
    before, after = raw['before']['metrics'], raw['after']['metrics']
    delta = {k: v - before.get(k, 0) for k, v in after.items()}
    row = dict(phase)
    drafts = total(delta, 'spec_decode_num_drafts_total')
    row['accepted_drafts_per_step'] = total(delta, 'spec_decode_num_accepted_tokens_total') / drafts if drafts else None
    row['uncached_prompt_tokens'] = row['prompt_tokens'] - row['cached_tokens']
    for metric in ('num_requests_running', 'num_requests_waiting'):
        weighted = coverage = 0.0
        peak = 0
        for left, right in zip(samples, samples[1:]):
            start = max(left['time'], phase['start'])
            end = min(right['time'], phase['end'])
            if end <= start or right['time'] - left['time'] > 25:
                continue
            value = total(left['metrics'], metric)
            weighted += value * (end - start)
            coverage += end - start
            peak = max(peak, value)
        row[metric] = {'time_weighted_mean': weighted / coverage if coverage else None,
                       'peak_sampled': peak, 'covered_seconds': coverage}
    rows.append(row)
groups = []
for enabled in (True, False):
    selected = [r for r in rows if r['gpu_readback'] == enabled]
    seconds = sum(r['seconds'] for r in selected)
    loaded = sum(r['load_bytes'] for r in selected)
    groups.append({'gpu_readback': enabled, 'windows': len(selected), 'seconds': seconds,
                   'aggregate_tps': sum(r['generated_tokens'] for r in selected) / seconds if seconds else None,
                   'load_seconds_per_GiB': sum(r['load_time'] for r in selected) / (loaded / 2**30) if loaded else None})
result = {'experiment_status': experiment['status'], 'phases': rows, 'groups': groups,
          'scope': 'Descriptive live windows. Context lengths, concurrency, cache and draft acceptance can vary. Summed transfer time is not wall-clock latency; no causal speedup established.'}
(out / 'analysis.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
