"""Compare the observed start of J with equal elapsed H/I3 workload windows."""
import json
import statistics
import time
from pathlib import Path

p = Path(__file__).resolve().parent
release = json.loads((p / 'J-release-requested.json').read_text())
all_rows = [json.loads(line) for line in (p / 'metrics.jsonl').read_text().splitlines()]
raw = (p / 'J-pre-resume-metrics.txt').read_text()
baseline = {'time': release['time'], 'config_sha256': release['config_sha256'],
            'metrics': {line.rsplit(' ', 1)[0]: float(line.rsplit(' ', 1)[1])
                        for line in raw.splitlines() if line.startswith('vllm:')
                        and '_bucket{' not in line and 'config_info{' not in line}}
j_rows = [baseline] + [r for r in all_rows if r.get('metrics')
                      and r.get('config_sha256') == release['config_sha256']
                      and r['time'] > release['time']]
assert len(j_rows) > 1
duration = j_rows[-1]['time'] - baseline['time']
groups = []
for label, index in [('H', 2), ('I3', 3)]:
    cycle = json.loads((p / f'batch-{index:03d}-counters.json').read_text())
    rows = [r for r in all_rows if r.get('metrics')
            and r.get('config_sha256') == cycle['config_sha256']
            and cycle['counter_start'] <= r['time'] <=
            min(cycle['counter_start'] + duration, cycle['counter_end'])]
    groups.append((label, rows))
groups.append(('J', j_rows))
output = []
for label, rows in groups:
    assert len(rows) > 1
    a, b = rows[0], rows[-1]
    def val(row, name):
        return sum(v for k, v in row['metrics'].items()
                   if k.startswith('vllm:' + name + '{'))
    def delta(name):
        value = val(b, name) - val(a, name)
        assert value >= 0, (label, name, 'counter reset')
        return value
    seconds = b['time'] - a['time']
    prompts = delta('prompt_tokens_total')
    drafts = delta('spec_decode_num_drafts_total')
    output.append({'config': label, 'start': a['time'], 'end': b['time'],
                   'seconds': seconds,
                   'generated_tokens': delta('generation_tokens_total'),
                   'aggregate_tps': delta('generation_tokens_total') / seconds,
                   'cache_fraction': delta('prompt_tokens_cached_total') / prompts if prompts else None,
                   'uncached_prompt_tokens': prompts - delta('prompt_tokens_cached_total'),
                   'completed_requests': delta('request_success_total'),
                   'mean_running_samples': statistics.mean(val(r, 'num_requests_running') for r in rows),
                   'accepted_drafts_per_attempt': delta('spec_decode_num_accepted_tokens_total') / drafts if drafts else None,
                   'preemptions': delta('num_preemptions_total'),
                   'payload_written': delta('kv_offload_store_bytes_total'),
                   'payload_loaded': delta('kv_offload_load_bytes_total')})
result = {'time': time.time(), 'rows': output,
          'scope': 'Descriptive matched elapsed workload starts, with metric sampling offsets. J starts immediately before controller release; actual launch delay must be reported. Different trajectories, cache state, concurrency and outputs; H includes readback trial/probes. No isolated causal estimate or full-cycle claim.'}
(p / 'H-I3-J-matched-elapsed.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
