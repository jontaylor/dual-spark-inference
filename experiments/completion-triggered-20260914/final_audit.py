import json,subprocess,shlex,hashlib,time,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1];result={'time':time.time(),'ranks':[]}
for rank in [0,1]:
 def run(args):return subprocess.check_output((['ssh','jon@192.168.100.11',shlex.join(args)] if rank else args),text=True,timeout=20)
 cfg=json.loads(run(['cat',str(root/'deploy_config.json')]));assert cfg==json.loads((p/f'candidate-events-r{rank}.json').read_text());assert cfg['mtp_tokens']==3
 container='qwen38-kv-paging-r'+str(rank);state=json.loads(run(['docker','inspect',container,'--format','{{json .State}}']));assert state['Running'] and not state['OOMKilled']
 for dst,h in cfg['runtime_override_sha256'].items():assert run(['docker','exec',container,'sha256sum','/usr/local/lib/python3.12/dist-packages/vllm/'+dst]).split()[0]==h,dst
 logs=run(['docker','logs','--since','2026-09-14T10:53:00Z',container]);(p/f'final-r{rank}.log').write_text(logs)
 result['ranks'].append({'rank':rank,'running':True,'started':state['StartedAt'],'mtp':3,'overrides_verified':len(cfg['runtime_override_sha256']),'config_sha256':hashlib.sha256(run(['cat',str(root/'deploy_config.json')]).encode()).hexdigest()})
with urllib.request.urlopen('http://127.0.0.1:30001/health',timeout=5) as f:result['health']=f.status
with urllib.request.urlopen('http://127.0.0.1:30001/metrics',timeout=5) as f:(p/'final-metrics.txt').write_bytes(f.read())
result['correctness_summaries']=[str(d/'summary.json') for d in p.iterdir() if d.is_dir() and d.name.startswith(('probe-','stop-')) and (d/'summary.json').exists() and json.loads((d/'summary.json').read_text()).get('passed')]
result['limitations']=['No universal cold-versus-cached numerical equality claim.','Pressure continuations passed; selected foreground targets stayed resident, so independent target-specific disk oracle coverage was not established.','Existing suspend-save path and transfer verification retained; no new forced suspension cycle.','Capacity samples are not matched historical workload replays.']
(p/'FINAL-AUDIT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
