"""Read-only source/configuration preflight. No deployment or model probes."""
import hashlib,json,subprocess,shlex,time
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1]
expected='aaf9452bafd005c945037888cecbe37bb2be5f77929ed811869074d0145035a4'
assert hashlib.sha256((root/'deploy_config.json').read_bytes()).hexdigest()==expected
assert json.loads((p/'validation-I3/summary.json').read_text())['passed']
assert json.loads((p/'I3-disk-validation.json').read_text())['passed']
rows=[]
for rank in (0,1):
 c=json.loads((p/f'candidate-J-reservation-r{rank}.json').read_text());b=json.loads((p/f'candidate-I3-mtp5-r{rank}.json').read_text());d=json.loads(json.dumps(c));assert d['kv_paging'].pop('reservation_tokens')==7680
 for key in ['runtime_overrides','runtime_override_sha256']:d[key].pop('v1/core/sched/gb10_parking_scheduler.py')
 assert d==b and c['kv_paging']['verify_transfers'] is True
 code="""import json,hashlib,sys;from pathlib import Path
c=json.load(sys.stdin)
for k,v in c['runtime_overrides'].items():assert hashlib.sha256(Path(v).read_bytes()).hexdigest()==c['runtime_override_sha256'][k],k
print(json.dumps({'source_hashes':len(c['runtime_overrides']),'launcher_hash':hashlib.sha256(Path('/home/jon/dual-spark-inference-kv-paging/launch_rank.py').read_bytes()).hexdigest(),'live_config_hash':hashlib.sha256(Path('/home/jon/dual-spark-inference-kv-paging/deploy_config.json').read_bytes()).hexdigest(),'live_config':json.loads(Path('/home/jon/dual-spark-inference-kv-paging/deploy_config.json').read_text())}))
"""
 cmd=['python3','-c',code];cmd=cmd if rank==0 else ['ssh','jon@192.168.100.11',shlex.join(cmd)]
 r=json.loads(subprocess.check_output(cmd,input=json.dumps(c),text=True,timeout=30));assert r.pop('live_config')==b and r['launcher_hash']==hashlib.sha256((root/'launch_rank.py').read_bytes()).hexdigest()
 code="""import json,hashlib,sys;from pathlib import Path
b=json.load(sys.stdin);root=Path('/usr/local/lib/python3.12/dist-packages/vllm')
for k,v in b['runtime_override_sha256'].items():assert hashlib.sha256((root/k).read_bytes()).hexdigest()==v,k
print('live override hashes verified')
"""
 cmd=['docker','exec','-i',f'qwen38-kv-paging-r{rank}','python3','-c',code];cmd=cmd if rank==0 else ['ssh','jon@192.168.100.11',shlex.join(cmd)]
 subprocess.run(cmd,input=json.dumps(b),text=True,check=True,timeout=30,capture_output=True)
 code="""import json,shutil,subprocess;from pathlib import Path
c=json.loads(Path('/home/jon/dual-spark-inference-kv-paging/deploy_config.json').read_text());cache=Path(c['kv_paging']['disk_root']);used=int(subprocess.check_output(['du','-sx','-B1',str(cache)],text=True).split()[0]);free=shutil.disk_usage(cache).free;needed=c['kv_paging']['disk_bytes_per_rank']+8*2**30;assert free+used>=needed,'Insufficient estimated disk after cache cleanup';print(json.dumps({'free_now':free,'allocated_cache_bytes':used,'estimated_cleanup_margin':free+used-needed}))
"""
 cmd=['python3','-c',code];cmd=cmd if rank==0 else ['ssh','jon@192.168.100.11',shlex.join(cmd)]
 disk=json.loads(subprocess.check_output(cmd,text=True,timeout=30));rows.append({'rank':rank,**r,'live_override_hashes':17,'disk_estimate':disk})
out={'time':time.time(),'passed':True,'ranks':rows,'scope':'Source/configuration only; requires fresh clean boundary before deployment and full live gates afterward.'};(p/'J-preflight.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
