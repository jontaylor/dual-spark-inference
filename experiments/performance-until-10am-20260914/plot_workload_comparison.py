"""Export descriptive workload curves; never infer a causal A/B speedup."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = Path(__file__).resolve().parent
data = [json.loads(line) for line in (p / 'metrics.jsonl').read_text().splitlines()]
release = json.loads((p / 'J-release-requested.json').read_text())
raw = (p / 'J-pre-resume-metrics.txt').read_text()
j_start = {'time': release['time'], 'metrics': {
    line.rsplit(' ', 1)[0]: float(line.rsplit(' ', 1)[1])
    for line in raw.splitlines() if line.startswith('vllm:')
    and '_bucket{' not in line and 'config_info{' not in line}}
j_rows = [j_start] + [r for r in data if r.get('metrics')
                       and r.get('config_sha256') == release['config_sha256']
                       and r['time'] > release['time']]
duration = j_rows[-1]['time'] - j_start['time']
series = []
for label, index in [('H · MTP3', 2), ('I3 · MTP5', 3)]:
    c = json.loads((p / f'batch-{index:03d}-counters.json').read_text())
    rows = [r for r in data if r.get('metrics')
            and r.get('config_sha256') == c['config_sha256']
            and c['counter_start'] <= r['time'] <=
            min(c['counter_start'] + duration, c['counter_end'])]
    series.append((label, rows))
series.append(('J · MTP5, smaller reservations', j_rows))

def metric(row, name):
    return sum(v for k, v in row['metrics'].items()
               if k.startswith('vllm:' + name + '{'))

plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False})
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for color, (label, rows) in zip(['#697485', '#d48721', '#007c83'], series):
    first = rows[0]
    xs, throughput, reuse, writes, loads = [], [], [], [], []
    for row in rows[1:]:
        seconds = row['time'] - first['time']
        if seconds < 60:
            continue
        def delta(name):
            value = metric(row, name) - metric(first, name)
            assert value >= 0, (label, name, 'counter reset')
            return value
        prompts = delta('prompt_tokens_total')
        xs.append(seconds / 60)
        throughput.append(delta('generation_tokens_total') / seconds)
        reuse.append(100 * delta('prompt_tokens_cached_total') / prompts if prompts else 0)
        writes.append(delta('kv_offload_store_bytes_total') / 1e9)
        loads.append(delta('kv_offload_load_bytes_total') / 1e9)
    for ax, values in zip(axes.flat, [throughput, reuse, writes, loads]):
        ax.plot(xs, values, label=label, color=color, linewidth=2)

for ax, title, units in zip(axes.flat,
                           ['Aggregate output throughput', 'Prompt-token reuse',
                            'New disk payload written', 'Disk payload loaded'],
                           ['tokens/s · cumulative average', '% · cumulative ratio',
                            'GB · cumulative', 'GB · cumulative']):
    ax.set_title(title, loc='left', fontweight='bold')
    ax.set_ylabel(units)
    ax.grid(alpha=.18)
    ax.set_ylim(bottom=0)
axes[0, 1].set_ylim(0, 100)
for ax in axes[1]:
    ax.set_xlabel('Minutes since workload start')
fig.suptitle(f'Observed workload starts · first {duration / 60:.1f} minutes',
             fontsize=16, fontweight='bold', y=.98)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94),
           ncol=3, frameon=False)
fig.text(.07, .025,
         'Descriptive comparison: trajectories, concurrency and cache state differ; H includes readback trials/probes.\n'
         'Throughput includes tool/client waits. Payload counters are not physical device I/O or unique recall rates.',
         fontsize=9, color='#4a5057')
fig.tight_layout(rect=[.02, .075, .99, .90])
fig.savefig(p / 'workload-comparison.png', dpi=160)
fig.savefig(p / 'workload-comparison.svg')
print(p / 'workload-comparison.png')
