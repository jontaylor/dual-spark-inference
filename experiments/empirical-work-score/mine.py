#!/usr/bin/env python3
"""Offline empirical work scoring. Reads completed campaign archives only.

Context is an HTTP-overlap estimate, NOT an observed GPU-resident context:
mean(prompt_tokens + output_tokens * request_progress), time-weighted over
each window. Queue and prefill time are included in request_progress.
Rates use actual server counter deltas. No serving processes are contacted.
"""
import argparse
import bisect
import csv
import datetime as dt
import gzip
import hashlib
import json
import math
from pathlib import Path
import tarfile


def stamp(s):
    return dt.datetime.fromisoformat(s).timestamp()


def counter(row, name):
    return sum(v for k, v in row['metrics'].items()
               if k.split('{')[0] == 'vllm:' + name)


def interpolate(rows, times, t, name):
    i = bisect.bisect_right(times, t) - 1
    if i < 0 or t > times[-1]:
        raise ValueError('counter endpoint outside recorded samples')
    if times[i] == t:
        return counter(rows[i], name)
    f = (t - times[i]) / (times[i + 1] - times[i])
    return counter(rows[i], name) * (1-f) + counter(rows[i+1], name) * f


def context(requests, lo, hi):
    active = [r for r in requests if r[0] < hi and r[1] > lo]
    boundaries = sorted({lo, hi} | {max(lo, r[0]) for r in active}
                        | {min(hi, r[1]) for r in active})
    area = occupied = count_area = 0.
    for a, b in zip(boundaries, boundaries[1:]):
        mid = (a+b)/2
        rr = [r for r in active if r[0] <= mid < r[1]]
        if rr:
            area += (b-a) * sum(r[2] + r[3]*(mid-r[0])/(r[1]-r[0]) for r in rr)/len(rr)
            occupied += b-a
            count_area += (b-a)*len(rr)
    return (area/occupied if occupied else None, occupied/(hi-lo), count_area/(hi-lo))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign', type=Path, default=Path(__file__).resolve().parents[1]/'scheduler-completion-20260915')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--window-seconds', type=int, default=30)
    ap.add_argument('--bucket-tokens', type=int, default=1000)
    ap.add_argument('--minutes', type=int, default=35, help='Cumulative scoring interval in minutes')
    ap.add_argument('--reference-minutes', type=int, default=0, help='Reference horizon; 0 uses all recorded data')
    args = ap.parse_args()
    if min(args.window_seconds, args.bucket_tokens, args.minutes) <= 0:
        ap.error('window, bucket and minutes must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    samples, sources, excluded = [], [], []
    for arm in sorted(args.campaign.glob('[0-9][0-9]-*')):
        archive = arm/'campaign-evidence.tar.gz'
        if not (arm/'complete.json').exists() or not archive.exists():
            excluded.append({'arm': arm.name, 'reason': 'No completed immutable archive'})
            continue
        w = json.loads((arm/'window.json').read_text())
        launch = stamp(w['controller_cycle_start'])
        requests, seen, invalid = [], set(), 0
        with tarfile.open(archive) as tar:
            for line in tar.extractfile('speculative-requests.jsonl'):
                r = json.loads(line)
                key = r.get('client_request_id')
                if key in seen:
                    continue
                seen.add(key)
                try:
                    u = r['usage']; start = stamp(r['started_utc']); end = stamp(r['ended_utc'])
                    prompt, output = u['prompt_tokens'], u['completion_tokens']
                    assert end > start and prompt >= 0 and output >= 0
                except (KeyError, TypeError, ValueError, AssertionError):
                    invalid += 1
                    continue
                requests.append((start, end, prompt, output))
        with gzip.open(arm/'metrics.jsonl.gz', 'rt') as f:
            rows = [json.loads(line) for line in f]
        times = [r['time'] for r in rows]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError(f'{arm.name}: non-increasing metric timestamps')
        names = ['generation_tokens_total', 'prompt_tokens_total', 'prompt_tokens_cached_total']
        for n in names:
            vals = [counter(r, n) for r in rows]
            if any(b < a for a, b in zip(vals, vals[1:])):
                raise ValueError(f'{arm.name}: counter reset: {n}')
        stop = min(launch + args.reference_minutes*60, times[-1]) if args.reference_minutes else times[-1]
        for offset in range(0, int(stop-launch), args.window_seconds):
            lo = launch+offset; hi = lo+args.window_seconds
            if lo < times[0] or hi > stop:
                continue
            c, coverage, count = context(requests, lo, hi)
            d = [interpolate(rows, times, hi, n)-interpolate(rows, times, lo, n) for n in names]
            prefill = d[1]-d[2]
            valid = c is not None and coverage >= .95 and prefill >= -1e-5
            samples.append(dict(arm=arm.name, start_seconds=offset, end_seconds=offset+args.window_seconds,
                context_estimate=c, bucket=int(c//args.bucket_tokens)*args.bucket_tokens if c is not None else None,
                context_coverage=coverage, mean_http_requests=count,
                generated_tokens=d[0], uncached_prompt_tokens=prefill,
                decode_tok_s=d[0]/args.window_seconds, prefill_tok_s=prefill/args.window_seconds,
                valid=valid))
        sources.append(dict(arm=arm.name, archive=str(archive.resolve()), requests=len(requests), invalid_requests=invalid,
            window=w, profiler_caveats=[json.loads(f.read_text()) for f in arm.glob('profile-*-intervention.json')]))
    if not samples:
        raise ValueError('No usable samples')
    peaks = {}
    for s in samples:
        if not s['valid']:
            continue
        b = s['bucket']
        if b not in peaks:
            peaks[b] = dict(bucket=b, samples=0, arms=set(), decode_peak=0., prefill_peak=0., decode_source='', prefill_source='')
        p = peaks[b]; p['samples'] += 1; p['arms'].add(s['arm'])
        for kind in ['decode', 'prefill']:
            if s[kind+'_tok_s'] > p[kind+'_peak']:
                p[kind+'_peak'] = s[kind+'_tok_s']
                p[kind+'_source'] = f"{s['arm']}:{s['start_seconds']}–{s['end_seconds']}s"
    for p in peaks.values():
        p['arms'] = ','.join(sorted(p['arms']))
    for s in samples:
        p = peaks.get(s['bucket']) if s['valid'] else None
        s['decode_units_s'] = s['decode_tok_s']/p['decode_peak'] if p and p['decode_peak'] else None
        s['prefill_units_s'] = s['prefill_tok_s']/p['prefill_peak'] if p and p['prefill_peak'] else None
        s['combined_units_s'] = s['decode_units_s']+s['prefill_units_s'] if s['decode_units_s'] is not None and s['prefill_units_s'] is not None else None
    scores = []
    for arm in sorted({s['arm'] for s in samples}):
        for label, lo, hi in [('0-35m' if args.minutes == 35 else f'0-{args.minutes}m', 0, args.minutes*60), ('30-35m', 1800, 2100)]:
            selected = [s for s in samples if s['arm']==arm and s['start_seconds']>=lo and s['end_seconds']<=hi]
            valid = [s for s in selected if s['combined_units_s'] is not None]
            scores.append(dict(arm=arm, interval=label, valid_seconds=len(valid)*args.window_seconds,
                observed_seconds=len(selected)*args.window_seconds,
                mean_units_s=sum(s['combined_units_s'] for s in valid)/len(valid) if valid else None,
                total_reference_units=sum(s['combined_units_s']*args.window_seconds for s in valid)))
    write_csv(args.output/'samples.csv', samples)
    write_csv(args.output/'peaks.csv', [peaks[b] for b in sorted(peaks)])
    write_csv(args.output/'scores.csv', scores)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for ax, kind in zip(axes, ['decode', 'prefill']):
        for arm in sorted({s['arm'] for s in samples}):
            ss = [s for s in samples if s['arm']==arm and s['valid']]
            ax.scatter([s['context_estimate']/1000 for s in ss], [s[kind+'_tok_s'] for s in ss], s=9, alpha=.4, label=arm)
        ax.plot([(b+args.bucket_tokens/2)/1000 for b in sorted(peaks)], [peaks[b][kind+'_peak'] for b in sorted(peaks)], color='black', linewidth=1.6, label='Maximum per context bucket')
        ax.set_ylabel(f'{kind.capitalize()} tokens/s'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xlabel('Estimated mean context of overlapping HTTP requests (thousands of tokens)')
    fig.suptitle(f'Empirical peaks: {args.window_seconds}s windows, all available reference windows\nContext reconstructed from request timing; throughput measured from server counters')
    fig.tight_layout(); fig.savefig(args.output/'curves.png', dpi=160); fig.savefig(args.output/'curves.svg'); plt.close(fig)
    manifest = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}, sources=sources, excluded=excluded,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), samples=len(samples), valid_samples=sum(s['valid'] for s in samples))
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    report = ['# Empirical work score', '', 'Rates are measured; context is estimated from overlapping HTTP requests, including queued requests. Output growth is assumed uniform across request lifetime. Requests without final usage/timing are excluded, so still-running calls at archival cutoff may be absent. No actual GPU context or token emission timing was recorded.', '',
        f'Reference: maximum observed {args.window_seconds}-second rate in each {args.bucket_tokens}-token context bucket, pooled over completed runs (reference horizon: {args.reference_minutes or 'all available'} minutes). Decode and prefill use the same mean context estimate. Peaks may come from different windows. Combined score = decode/peak_decode + uncached_prefill/peak_prefill. This is an empirical index, not physical GPU utilisation; values above 1 are allowed.', '',
        'Only windows with at least 95% reconstructed-context coverage and nonnegative counter deltas are scored. Missing or zero denominators give an unknown score. Scores average only covered windows; inspect valid_seconds before comparing. Sparse buckets and winners setting their own reference remain visible in peaks.csv. Reference changes when the source cohort changes. Lines connect occupied bucket centres for display; scoring uses the bucket maxima directly, without interpolation.', '',
        '| Run | Interval | Scored seconds | Mean units/s |', '|---|---|---:|---:|']
    for s in scores:
        report.append(f"| {s['arm']} | {s['interval']} | {s['valid_seconds']} | {s['mean_units_s']:.4f} |" if s['mean_units_s'] is not None else f"| {s['arm']} | {s['interval']} | 0 | UNKNOWN |")
    report += ['', '![Empirical curves](curves.png)', '', 'Raw data: samples.csv, peaks.csv, scores.csv. Source paths, request exclusions and profiler caveats: manifest.json. No live services or campaign files were modified.']
    (args.output/'REPORT.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'output':str(args.output), 'arms':len(sources), 'samples':len(samples), 'valid_samples':manifest['valid_samples']}, indent=2))


if __name__ == '__main__':
    main()
