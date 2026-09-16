"""Read-only source analysis; writes derived results beside the raw campaign evidence."""
import ast,collections,datetime,gzip,json,pathlib,statistics,tarfile
P=pathlib.Path(__file__).resolve().parent
# Reuse the counter summarizer without running the old ten-minute analysis.
src=ast.parse((P.parent/'scheduler-campaign-20260915/analyze.py').read_text())
exec(compile(ast.Module(body=[n for n in src.body if isinstance(n,ast.FunctionDef) and n.name in ('aggregate','summarize')],type_ignores=[]),'counter_helpers','exec'))
results=[]
measurement_caveats=[{'arm':f.parent.name,'source':str(f.relative_to(P)),**json.loads(f.read_text())} for f in sorted(P.glob('[0-9][0-9]-*/profile-*-intervention.json'))]
for arm in sorted(P.glob('[0-9][0-9]-*')):
 if not (arm/'complete.json').exists():continue
 w=json.loads((arm/'window.json').read_text());done=json.loads((arm/'complete.json').read_text())
 cut=json.loads((arm/'cutoff.json').read_text());snap=json.loads((arm/'completion-cutoff.json').read_text())
 history=[json.loads(l) for l in (arm/'completion-history.jsonl').read_text().splitlines()]
 count=lambda h:sum(r['status']=='complete' for r in h['rows'])
 assert len(snap['rows'])==16 and count(snap)>=8 and done['same_campaign']
 assert all(h['campaign']==w['campaign'] for h in history)
 first=next(h for h in history if count(h)>=8)
 prior=[h for h in history if h['observed_time']<first['observed_time']]
 launch=datetime.datetime.fromisoformat(w['controller_cycle_start']).timestamp()
 lower=prior[-1]['observed_time'] if prior else w['start']
 metrics=[json.loads(l) for l in gzip.open(arm/'metrics.jsonl.gz','rt')];metrics.append(cut)
 full=summarize(metrics,w['start'],cut['time'])
 eighth_metrics=summarize(metrics,w['start'],first['observed_time'])
 assert all(v>=0 for v in full['deltas'].values()),'Counter reset in measurement'
 quality=[]
 with tarfile.open(arm/'campaign-evidence.tar.gz') as t:
  for member in t:
   if member.name.count('/')==1 and member.name.endswith('/run.json'):
    d=json.load(t.extractfile(member))
    quality.append({'id':member.name.split('/')[0],**{k:v for k,v in d.items() if k not in ('calls','tool_events','filter_events','role_settings') and any(s in k for s in ('status','check','test','error','elapsed','stop_reason','exclusion','nonpass'))}})
 results.append({'arm':arm.name,'measurement_caveats':[c for c in measurement_caveats if c['arm']==arm.name],'time_to_eighth_from_launch_seconds':first['observed_time']-launch,'crossing_interval_from_launch_seconds':[lower-launch,first['observed_time']-launch],'time_from_first_inference_seconds':first['observed_time']-w['start'],'completed_at_cutoff':count(snap),'task_snapshot':snap,'quality_at_export':quality,'full':full,'through_eighth':eighth_metrics,'completion_target':w.get('completion_target',8),'time_to_target_from_launch_seconds':snap['observed_time']-launch,'window':w,'cutoff_time':cut['time']})
(P/'results.json').write_text(json.dumps(results,indent=2)+'\n')
scope=json.loads((P/'scope-revision.json').read_text()) if (P/'scope-revision.json').exists() else None
matrix=json.loads((P/'matrix.json').read_text())
required=[n for n,(s,b,t) in enumerate(matrix,1) if scope is None or (s not in scope['excluded_max_num_seqs'] and t not in scope['excluded_thresholds'])]
policy=json.loads((P/'final-run-policy.json').read_text()) if (P/'final-run-policy.json').exists() else {}
required=policy.get('required_arm_ids',required)
measured_required=sum(int(r['arm'].split('-')[0]) in required for r in results)
lines=['# Completion campaign results','',f'{measured_required} / {len(required)} currently requested configurations complete; {len(results)} total historical measurements retained. Primary endpoint: eighth normally completed task; this is not proof of quality-test success. Crossing intervals reflect polling.','', '| Configuration | Time to eighth from launch (s) | Polling interval (s) | Generated tok/s | Uncached prompt tok/s | Prompt reuse % |','|---|---:|---:|---:|---:|---:|']
if scope:lines.insert(2,'Scope revised by user: skip all s8, s24 and t512. Historical completed results below are retained; skipped partial runs are not measurements. See scope-revision.json.')
for r in results:
 f=r['through_eighth'];d=f['deltas'];uncached=(d.get('vllm:prompt_tokens_total',0)-d.get('vllm:prompt_tokens_cached_total',0))/f['sampled_seconds'];reuse=f['cached_prompt_fraction'];lo,hi=r['crossing_interval_from_launch_seconds']
 lines.append(f"| {r['arm']} | {hi:.1f} | {lo:.1f}–{hi:.1f} | {f['generation_tok_s']:.2f} | {uncached:.2f} | {100*reuse:.2f} |" if reuse is not None else f"| {r['arm']} | {hi:.1f} | {lo:.1f}–{hi:.1f} | {f['generation_tok_s']:.2f} | {uncached:.2f} | UNKNOWN |")
lines+=['','Full counter and label deltas, task identities and quality evidence are in results.json. See [QUALITY.md](QUALITY.md) for benchmark, original CLI and supplementary verification of cutoff finishers. Reuse includes GPU-resident completion restores; external labels are not equivalent to disk reads. Different completed task subsets and trajectories still limit causal comparisons.']
for r in results:
 if r['completion_target']!=8:
  lines+=['',f"Final run {r['arm']}: {r['completed_at_cutoff']} tasks completed in {r['time_to_target_from_launch_seconds']:.1f}s from launch; full-run generation {r['full']['generation_tok_s']:.2f} tok/s. Comparison table uses the first-eight window; full counters are separate in results.json."]
if measurement_caveats:
 lines+=['','## Measurement caveats','']
 for c in measurement_caveats:
  lines.append(f"- **{c['arm']}**: {c['reason']}. {c['validity']} See [{c['source']}]({c['source']}).")
failures=sorted(P.glob('[0-9][0-9]-*/failed.json'))
if failures:
 lines+=['','## Failed attempts — not completed measurements','']
 for path in failures:
  failure=json.loads(path.read_text())
  lines.append(f"- **{path.parent.name}**: {failure['outcome']}. {failure['cause']}. See [{path.parent.name}/failed.json]({path.parent.name}/failed.json).")
(P/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
