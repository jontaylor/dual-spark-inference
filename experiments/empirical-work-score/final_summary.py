"""Build the final comparison from frozen smooth-final outputs and campaign evidence."""
import csv,json,pathlib,datetime,itertools,runpy
P=pathlib.Path(__file__).resolve().parent
C=P.parent/'scheduler-completion-20260915';O=P/'smooth-final'
rs={r['arm']:r for r in json.loads((C/'results.json').read_text())}
quality={r['arm']:r for r in json.loads((C/'quality-audit.json').read_text())['arms']}
scores=[r for r in csv.DictReader((O/'scores.csv').open()) if r['interval']=='0-35m']
windows=list(csv.DictReader((O/'scored-windows.csv').open()))
f=runpy.run_path(str(P/'fit_functions.py'))['WorkFunctions'](O/'functions.json')
for r in windows:
 c=float(r['context']);expected=f.decode_units(c,float(r['decode_tok_s']))+f.prefill_units(c,float(r['prefill_tok_s']));assert abs(expected-float(r['combined_units_s']))<1e-10
first={};quals={}
for name,r in rs.items():
 launch=datetime.datetime.fromisoformat(r['window']['controller_cycle_start']).timestamp();history=[json.loads(l) for l in (C/name/'completion-history.jsonl').open()]
 first[name]=next(h['observed_time']-launch for h in history if any(t['status']=='complete' for t in h['rows']))
 h=next(h for h in history if sum(t['status']=='complete' for t in h['rows'])>=8);ids={t['id'] for t in h['rows'] if t['status']=='complete'}
 checks=[t['verification'].get('supplementary_passed') for t in quality[int(name[:2])]['tasks'] if t['id'] in ids];assert len(checks)==8
 quals[name]=f'{sum(x is True for x in checks)}/{sum(x is False for x in checks)}/{sum(x is None for x in checks)}'
ordered=sorted(scores,key=lambda r:-float(r['combined_units_s']))
lines=['# Final scheduler work analysis','',
 'The campaign has ended under its revised scope. There are 14 completed historical measurements; the final six requested configurations have measurements. Two s32/b32768 attempts ended under memory pressure and were not retried. This report performs offline analysis only; no services or controller state were changed.','',
 '**Conclusion:** run 3 (s32/b8192/t1024) leads the context-adjusted first-35-minute work score. The predicted s32/b16384/t1024 improvement did not materialise. This is the strongest measured candidate under the chosen score, not a statistically established optimum.','',
 'All 14 runs were refitted and rescored against a common 90th-percentile empirical reference. Scores are not directly comparable in absolute scale to previous report versions. The reference uses all archived windows; run 36 alone continued until all 16 tasks finished. Its longer tail is included in reference fitting, with equal total fitting weight per run.','',
 '| Work rank | Run / settings | Decode units/s | Prefill units/s | Combined | Scored seconds | Time to 8 | Supplementary pass/fail/unknown, first 8 |',
 '|---:|---|---:|---:|---:|---:|---:|---|']
for i,s in enumerate(ordered,1):
 name=s['arm'];secs=rs[name]['time_to_eighth_from_launch_seconds'];mm=int(secs//60);ss=secs%60
 lines.append(f"| {i} | {name} | {float(s['decode_units_s']):.3f} | {float(s['prefill_units_s']):.3f} | {float(s['combined_units_s']):.3f} | {s['scored_seconds']} | {mm}m {ss:04.1f}s | {quals[name]} |")
lines+=['','## Comparisons changing one setting','', 'These are observed differences in single runs, not causal parameter estimates. Changing one setting still changed the model trajectory.','', '| Change | Other settings | Score change |','|---|---|---:|']
lookup={tuple(map(int,[s['arm'].split('-')[1][1:],s['arm'].split('-')[2][1:],s['arm'].split('-')[3][1:]])):float(s['combined_units_s']) for s in scores}
for dim,label in [(0,'sequences'),(1,'budget'),(2,'threshold')]:
 for a,b in itertools.combinations(sorted(lookup),2):
  if sum(x!=y for x,y in zip(a,b))==1 and a[dim]!=b[dim]:
   other=', '.join(f'{k}={a[j]}' for j,k in enumerate(['s','b','t']) if j!=dim)
   lines.append(f'| {label} {a[dim]} → {b[dim]} | {other} | {(lookup[b]/lookup[a]-1)*100:+.1f}% |')
lines+=['','## Robustness and coverage','',
 'The first-35-minute score includes 2070 seconds (00:30–35:00) for every run except run 19, which has 2040. Its missing valid window must not be silently counted as zero. Runs 12 and 35 had first observed task completions at 33.03m and 31.93m respectively. Thus the old pre-task-completion assumption does not hold for the entire expanded cohort.','',
 'The following comparisons use identical recorded reference functions; no reference refit is done for the shorter interval. All runs have the complete 00:30–30:00 interval scored (1770 seconds), preceding all first observed whole-task completions.','', '| Check | Best → worst run IDs |','|---|---|']
for key,label in [('combined_q0.85','35m, 85th reference'),('combined_units_s','35m, 90th reference'),('combined_q0.95','35m, 95th reference')]:
 order=sorted(scores,key=lambda r:-float(r[key]));lines.append('| '+label+' | '+' → '.join(str(int(r['arm'][:2])) for r in order)+' |')
short=[]
for s in scores:
 ww=[r for r in windows if r['arm']==s['arm'] and int(r['start_seconds'])>=30 and int(r['end_seconds'])<=1800];duration=sum(int(r['end_seconds'])-int(r['start_seconds']) for r in ww);assert duration==1770
 avg=sum(float(r['combined_units_s'])*(int(r['end_seconds'])-int(r['start_seconds'])) for r in ww)/duration;short.append((s['arm'],avg,duration))
lines.append('| 00:30–30:00, 90th reference | '+' → '.join(str(int(a[:2])) for a,v,d in sorted(short,key=lambda r:-r[1]))+' |')
lines+=['','| Run | Mean units/s, 00:30–30:00 |','|---|---:|']
for a,v,d in sorted(short,key=lambda r:-r[1]):lines.append(f'| {a} | {v:.3f} |')
lines+=['','## Completion, quality and failures','',
 'Fastest time to eight completions: run 12, s32/b16384/t2048, 47m08.1s. Its first eight finishers passed only 1/8 supplementary checks. This is an end-to-end completion result and is not interchangeable with the computational work score.','',
 'Run 36, s32/b16384/t1024, reached eight at 68m52.3s and all 16 at 85m28.4s. Its first eight passed 1/8 supplementary checks; all 16 passed 5/16. All completed tasks passed the benchmark and original CLI checks, but supplementary failures remain. The original campaign QUALITY.md uses all 16 for run 36; the comparison table above consistently uses the first eight for every run.','',
 'Both s32/b32768 attempts failed with supervisor memory-pressure shutdown: run 13 at t2048 (MemAvailable 5.16 GiB, swap 16.0 GiB), run 34 at t1024 (MemAvailable 7.96 GiB, swap 15.835 GiB). They have no completed work-score measurement here. They are not demonstrated viable settings under the tested workload. Historical s24/b32768 did complete.','',
 'Run 9 has a severe profiler overrun. Run 36 has a smaller sampling-lag caveat (1.15s behind, 29 sampling errors). These remain documented in the mined manifest and campaign report.','',
 '## Interpretation and limits','',
 'Sequence count: s8 is consistently poor in this workload. s32 versus s24 is not uniformly positive: it helps at b8192, but the b16384/t1024 comparison favours s24. Budget: 16k is not a universal improvement; s32/t1024 peaks at 8k while s32/t2048 peaks at 16k among completed runs. Threshold: at s32, t1024 wins at 4k and 8k budgets, while t2048 wins at 16k. These reversals rule out multiplying independent parameter gains to predict the optimum.','',
 'These are empirical operation-rate units, not physical GPU utilisation or useful-task units. Context is an HTTP-overlap estimate, including queues and uniform output growth. Reprocessing earns computation credit. Fits are retrospective, use mixed operation windows and have no prospective repeated-run validation. Different trajectories, cache behaviour, concurrency and quality remain confounds. Do not call any configuration fully deterministic or fully quality-verified based on this campaign.','',
 'Recommendation under the user’s preferred work criterion: s32/b8192/t1024 is the measured leader to carry forward. s24/b16384/t1024 is a close historical alternative. Retain the distinction from the end-to-end completion leader and avoid a deployment change based solely on this report.','',
 '## Artifacts and reproduction','',
 '- [Smooth fits and detailed report](REPORT.md)',
 '- [Scores](scores.csv)',
 '- [Per-window calculations](scored-windows.csv)',
 '- [Reference coefficients and validation](functions.json)',
 '- [Mining manifest](../mined-final/manifest.json)',
 '- [Handover and full methodology](../HANDOVER.md)','',
 '```bash\ncd /home/jon/dual-spark-inference-kv-paging\npython3 experiments/empirical-work-score/mine.py --campaign experiments/scheduler-completion-20260915 --window-seconds 30 --bucket-tokens 1000 --minutes 35 --reference-minutes 0 --output experiments/empirical-work-score/mined-final-reproduction\npython3 experiments/empirical-work-score/fit_functions.py --samples experiments/empirical-work-score/mined-final-reproduction/samples.csv --quantile 0.90 --output experiments/empirical-work-score/smooth-final-reproduction\n```','',
 'Output directories must be new. `final_summary.py` regenerates this additional synthesis using the fixed smooth-final and campaign paths. Arithmetic validation recomputed all exported scores, and checked complete common-window coverage. Source campaign audit reports all 14 completed measurement protocols passed; that does not certify quality or determinism.']
(O/'FINAL_ANALYSIS.md').write_text('\n'.join(lines)+'\n')
(O/'final-validation.json').write_text(json.dumps(dict(verified_window_scores=len(windows),first_completion_seconds=first,common_30m_scores=short,first_eight_supplementary=quals),indent=2)+'\n')
print('\n'.join(lines[:24]));print('SHORT',sorted(short,key=lambda r:-r[1]))
