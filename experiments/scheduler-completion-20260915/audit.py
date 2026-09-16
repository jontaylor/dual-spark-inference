"""Audit completed measurement arms against the full requested matrix and raw cutoffs."""
import itertools,json,pathlib,tarfile
P=pathlib.Path(__file__).resolve().parent
matrix=json.loads((P/'matrix.json').read_text())
assert len(matrix)==36 and set(map(tuple,matrix))==set(itertools.product((8,24,32),(4096,8192,16384,32768),(512,1024,2048)))
scope=json.loads((P/'scope-revision.json').read_text()) if (P/'scope-revision.json').exists() else None
required=[n for n,(s,b,t) in enumerate(matrix,1) if scope is None or (s not in scope['excluded_max_num_seqs'] and t not in scope['excluded_thresholds'])]
policy=json.loads((P/'final-run-policy.json').read_text()) if (P/'final-run-policy.json').exists() else {}
required=policy.get('required_arm_ids',required)
base=[json.loads((P/f'baseline-r{r}.json').read_text()) for r in (0,1)]
results=[];reference=None;quality=[]
for n,(s,b,t) in enumerate(matrix,1):
 p=P/f'{n:02d}-s{s}-b{b}-t{t}'
 if not (p/'complete.json').exists():continue
 checks={}
 for rank in (0,1):
  cfg=json.loads((p/f'config-r{rank}.json').read_text());expected={**base[rank], 'max_num_seqs':s,'max_num_batched_tokens':b,'long_prefill_token_threshold':t}
  checks[f'rank{rank}_only_requested_settings_changed']=cfg==expected
  live=json.loads((p/f'runtime-r{rank}.json').read_text());args=live['args']
  checks[f'rank{rank}_actual_arguments']=all(args[args.index(flag)+1]==str(value) for flag,value in (('--max-num-seqs',s),('--max-num-batched-tokens',b),('--long-prefill-token-threshold',t)))
  checks[f'rank{rank}_running_at_capture']=live['state']['Running']
 done=json.loads((p/'complete.json').read_text());w=json.loads((p/'window.json').read_text());snap=json.loads((p/'completion-cutoff.json').read_text())
 hist=[json.loads(l) for l in (p/'completion-history.jsonl').read_text().splitlines()]
 count=lambda h:sum(r['status']=='complete' for r in h['rows'])
 checks['same_campaign']=done['same_campaign'] and done['campaign_at_start']==done['campaign_at_end']==w['campaign']==snap['campaign']
 checks['16_distinct_tasks']=len(snap['rows'])==len({r['id'] for r in snap['rows']})==16
 target=policy.get('completion_targets',{}).get(str(n),8)
 checks['requested_completions']=count(snap)>=target and done['completed']==count(snap) and done.get('completion_target',8)==w.get('completion_target',8)==target
 checks['stopped_at_first_observed_crossing']=count(hist[-1])>=target and all(count(h)<target for h in hist[:-1]) and hist[-1]==snap
 checks['cutoff_after_completion_observation']=json.loads((p/'cutoff.json').read_text())['time']>=snap['observed_time']
 with tarfile.open(p/'campaign-evidence.tar.gz') as archive:
  manifest=json.load(archive.extractfile('campaign.json'))
  conditions=[{**c,'source':'NORMALIZED_CAMPAIGN_SOURCE'} for c in manifest['conditions']]
  frozen={k:manifest.get(k) for k in ('parent_campaign','inference_slots_per_arm','output_limits','request_concurrency_ceiling','continuous')};frozen['conditions']=conditions
  if reference is None:reference=frozen
  checks['same_workload_spec']=frozen==reference
  checks['temperature_zero']=all(c['sampling_temperature']==0 for c in conditions)
  checks['same_16_task_ids']=set(c['id'] for c in conditions)==set(r['id'] for r in snap['rows'])
  names=set(archive.getnames());task_quality=[]
  for row in snap['rows']:
   if row['status']!='complete':continue
   name=row['id']+'/operator-audit/verification.json'
   verification=json.load(archive.extractfile(name)) if name in names else None
   detail_name=row['id']+'/operator-audit/relative-path-cwd.json'
   detail=json.load(archive.extractfile(detail_name)) if detail_name in names else None
   task_quality.append({'id':row['id'],'verification_source':name if verification is not None else None,'verification':verification,'relative_path_cwd':detail})
  counts={key:{'pass':sum(x['verification'] is not None and x['verification'].get(key) is True for x in task_quality),'fail':sum(x['verification'] is not None and x['verification'].get(key) is False for x in task_quality),'unknown':sum(x['verification'] is None or x['verification'].get(key) not in (True,False) for x in task_quality)} for key in ('benchmark_passed','original_cli_passed','supplementary_passed','verified_repair')}
  quality.append({'arm':n,'configuration':[s,b,t],'completed_tasks':len(task_quality),'counts':counts,'tasks':task_quality})
 results.append({'arm':n,'checks':checks,'passed':all(checks.values())})
report={'matrix_complete_and_unique':True,'measured_arms':len(results),'required_arms':len(required),'required_arm_ids':required,'measured_required_arms':sum(r['arm'] in required for r in results),'scope_revision':scope,'final_run_policy':policy,'all_completed_arms_pass':all(r['passed'] for r in results),'campaign_complete':all(any(r['arm']==n and r['passed'] for r in results) for n in required),'arms':results,'limits':'Does not certify model numerical determinism or task quality; inspect separate quality evidence.'}
report['failed_attempts']=[{'arm_directory':f.parent.name,**json.loads(f.read_text())} for f in sorted(P.glob('[0-9][0-9]-*/failed.json'))]
(P/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
(P/'quality-audit.json').write_text(json.dumps({'scope':'Only tasks complete at the measurement cutoff. Archived operator audit results; benchmark pass does not mean the full test suite passed. Missing evidence is UNKNOWN. Supplementary failures are distinct from known environment exclusions. Does not certify model numerical determinism.','arms':quality},indent=2)+'\n')
lines=['# Quality of tasks completed at cutoff','','Counts are pass / fail / unknown from archived operator verification. Completion remains the requested stopping criterion. These checks do not establish numerical determinism or a clean full test suite.','','| Arm | Sequences / budget / threshold | Benchmark | Original CLI | Supplementary | Verified repair |','|---|---|---|---|---|---|']
for arm in quality:
 cells=[' / '.join(str(arm['counts'][key][state]) for state in ('pass','fail','unknown')) for key in ('benchmark_passed','original_cli_passed','supplementary_passed','verified_repair')]
 lines.append('| '+str(arm['arm'])+' | '+' / '.join(map(str,arm['configuration']))+' | '+' | '.join(cells)+' |')
lines+=['','Task identities, source paths and relative-path/CWD check details are preserved in [quality-audit.json](quality-audit.json). Different finishing subsets and a single run per configuration prevent attributing these quality differences to scheduler settings.']
(P/'QUALITY.md').write_text('\n'.join(lines)+'\n')
