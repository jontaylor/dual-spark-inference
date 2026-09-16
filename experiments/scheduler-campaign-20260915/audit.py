import pathlib,json,gzip,tarfile,statistics
P=pathlib.Path(__file__).resolve().parent;matrix=json.loads((P/'matrix.json').read_text());base=[json.loads((P/f'baseline-r{r}.json').read_text()) for r in (0,1)];reference=None;results=[]
for n,(seq,budget,threshold) in enumerate(matrix,1):
 p=P/f'{n:02d}-s{seq}-b{budget}-t{threshold}'
 if not (p/'complete.json').exists():results.append({'arm':n,'complete':False});continue
 checks={}
 for r in (0,1):
  c=json.loads((p/f'config-r{r}.json').read_text());expected=dict(base[r]);expected.update(max_num_seqs=seq,max_num_batched_tokens=budget,long_prefill_token_threshold=threshold);checks[f'config_scope_r{r}']=c==expected
  runtime=json.loads((p/f'runtime-r{r}.json').read_text());a=runtime['args'];checks[f'actual_args_r{r}']=all(a[a.index(flag)+1]==str(v) for flag,v in [('--max-num-seqs',seq),('--max-num-batched-tokens',budget),('--long-prefill-token-threshold',threshold)])
  checks[f'runtime_running_r{r}']=runtime['state']['Running']
 w=json.loads((p/'window.json').read_text());end=json.loads((p/'cutoff.json').read_text());checks['ten_minute_window']=600<=end['time']-w['start']<610
 rows=[json.loads(l) for l in gzip.open(p/'metrics.jsonl.gz','rt')];selected=[r for r in rows if w['start']<=r['time']<=w['start']+600];gaps=[b['time']-a['time'] for a,b in zip(selected,selected[1:])];checks['metrics_coverage']=selected[-1]['time']-selected[0]['time']>=590 and max(gaps)<10
 hw=[json.loads(l) for l in gzip.open(p/'hardware.jsonl.gz','rt')];checks['hardware_both_ranks']=set(r['rank'] for r in hw if 'data' in r)=={0,1};checks['hardware_no_errors']=not any('error' in r for r in hw)
 checks['same_fresh_campaign']=json.loads((p/'complete.json').read_text())['same_campaign']
 for offset in (120,420):
  for label in ('engine','worker'):
   f=p/f'profile-{offset}-{label}.json';checks[f'profile_{offset}_{label}']=f.exists() and bool(json.loads(f.read_text()).get('profiles'))
 with tarfile.open(p/'campaign-evidence.tar.gz') as t:
  campaign=json.load(t.extractfile('campaign.json'));source=json.load(t.extractfile('current-source-sha256.json'));conditions=[{k:v for k,v in c.items() if k!='source'} for c in campaign['conditions']]
  frozen={'source':source,'conditions':conditions,'output_limits':campaign['output_limits'],'slots':campaign['inference_slots_per_arm'],'active':campaign['active_conditions'],'ceiling':campaign['request_concurrency_ceiling']}
  if reference is None:reference=frozen
  checks['frozen_workload_matches']=frozen==reference
  checks['campaign_archive_readable']=len(t.getmembers())>0
 results.append({'arm':n,'complete':True,'checks':checks,'all_checks_pass':all(checks.values()),'samples':len(selected),'max_metrics_gap_s':max(gaps),'window_seconds':end['time']-w['start']})
report={'all_eight_collected':all(x['complete'] for x in results),'all_checks_pass':all(x.get('all_checks_pass',False) for x in results),'arms':results,'scope':'Campaign configuration, collection and workload-specification audit; not model numerical determinism or task-quality validation.'}
(P/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({'all_eight_collected':report['all_eight_collected'],'arms':[{k:v for k,v in x.items() if k!='checks'} for x in results],'failures':{x['arm']:[k for k,v in x.get('checks',{}).items() if not v] for x in results if x.get('checks') and not x['all_checks_pass']}},indent=2))
