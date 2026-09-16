"""Validate an already-ready, idle candidate; unique evidence paths, no restart."""
import hashlib,json,subprocess,sys,time,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1];prior=p.parent/'aggregate-throughput-20260914';spec=p.parent/'spec-determinism-20260914';py=str(root/'.venv/bin/python')
label=sys.argv[1]
assert label and all(c.isalnum() or c=='-' for c in label)
out=p/('validation-'+label);out.mkdir(exist_ok=False)
record={'started':time.time(),'label':label,'config_sha256':hashlib.sha256((root/'deploy_config.json').read_bytes()).hexdigest(),'gates':{},'scope':'Within-config tested cache and C1/C4/mixed routes. No unconditional cold-vs-cached or cross-restart equality claim.'}
def save():(out/'summary.json').write_text(json.dumps(record,indent=2))
def run(script,*args):
 with (out/(script.stem+'-'+('-'.join(args) or 'run')+'.log')).open('w') as log:subprocess.run([py,str(script),*args],stdout=log,stderr=subprocess.STDOUT,check=True)
def idle():
 with urllib.request.urlopen('http://127.0.0.1:30001/metrics',timeout=5) as r:raw=r.read().decode()
 gauges=[l for l in raw.splitlines() if l.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))];assert len(gauges)>=2 and all(float(l.rsplit(' ',1)[1])==0 for l in gauges),'Server not confirmed idle'
try:
 idle();save();mode='after-'+label;run(prior/'cache_probe.py',mode);r=json.loads((prior/mode/'summary.json').read_text());assert r['warm_parallel_tokens_equal'] and r['warm_parallel_scores_equal'] and r['warm_cache']['cached_tokens']==r['expected_boundary'];record['gates']['cache']={'path':str(prior/mode),'summary':r};save()
 for suffix in ('','-seed'):
  mode='full-'+label+suffix;idle();run(spec/'probe.py',mode);rows=json.loads((spec/mode/'results.json').read_text());assert len(rows)==10
  ref=rows[0];assert ref['prompt_token_ids'] and ref['token_ids'] and ref['logprobs'],'Missing diagnostics'
  assert all(r['error'] is None and all(r[k]==ref[k] for k in ('prompt_token_ids','token_ids','logprobs','canonical')) for r in rows),'Concurrent/serial divergence'
  record['gates'][mode]={'path':str(spec/mode),'requests':10,'generated_tokens_each':len(ref['token_ids']),'exact':True};save()
 idle();before=set(spec.glob('mixed-*'));run(spec/'mixed_probe.py');created={x for x in set(spec.glob('mixed-*'))-before if x.is_dir()};assert len(created)==1;folder=created.pop();rows=json.loads((folder/'summary.json').read_text());assert len(rows)==12 and all(all(r[k] for k in ('tokens_equal','scores_equal','prompt_equal')) for r in rows)
 samples=[json.loads(l) for l in (folder/'metrics.jsonl').read_text().splitlines()];peak=max((sum(float(v.rsplit(' ',1)[1]) for v in s.get('metrics',[]) if v.startswith('vllm:num_requests_running{')) for s in samples),default=0);assert peak>=2,'No observed concurrency coverage'
 record['gates']['mixed']={'path':str(folder),'comparisons':12,'exact':True,'observed_peak_running':peak,'label':'8 submitted clients; actual overlap reported separately'};record['passed']=True
except BaseException as e:record['passed']=False;record['error']=repr(e);raise
finally:record['finished']=time.time();save();print(json.dumps(record,indent=2),flush=True)
