"""Exercise candidate launch command generation against current local resources.

Uses --dry-run, temporary merged config and the repository's actual path roots.
Never changes deploy_config.json or invokes docker run/rm.
"""
import contextlib,copy,io,json,os,sys,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
rank=int(sys.argv[1]) if len(sys.argv)>1 else 0
assert rank in (0,1)
sys.path.insert(0,str(ROOT))
base=json.loads((ROOT/'deploy_config.json').read_text())
manifest=json.loads((P/'PATCH_MANIFEST.json').read_text())
results=[]
for enabled in (False,True):
 cfg=copy.deepcopy(base);cfg['async_scheduling']=enabled
 cfg['kv_paging']['async_terminal_snapshot_bytes']=2**31
 for entry in manifest['replacements']:
  if entry['target']=='launch_rank.py':continue
  cfg['runtime_overrides'][entry['target']]=str(ROOT/entry['candidate'])
  cfg['runtime_override_sha256'][entry['target']]=entry['candidate_sha256']
 with tempfile.NamedTemporaryFile('w',suffix='.json') as f:
  json.dump(cfg,f);f.flush()
  os.environ['DUAL_SPARK_CONFIG']=f.name
  sys.argv=[str(ROOT/'launch_rank.py'),str(rank),'--dry-run']
  stream=io.StringIO()
  with contextlib.redirect_stdout(stream):
   exec(compile((P/'candidate/launch_rank.py').read_text(),str(ROOT/'launch_rank.py'),'exec'),
        {'__name__':'__main__','__file__':str(ROOT/'launch_rank.py')})
 outputs=[json.loads(line) for line in stream.getvalue().splitlines()]
 args=outputs[0]['args'];cmd=outputs[1]['docker_command']
 flag='--async-scheduling' if enabled else '--no-async-scheduling'
 assert flag in args
 expected=base['kv_paging']['kv_bytes_per_rank']-(2**31 if enabled else 0)
 assert int(args[args.index('--kv-cache-memory')+1])==expected
 assert args[args.index('--scheduler-cls')+1].endswith('GB10AsyncParkingScheduler' if enabled else 'GB10ParkingScheduler')
 for entry in manifest['replacements']:
  if entry['target']=='launch_rank.py':continue
  assert any(entry['candidate'] in word for word in cmd)
 results.append(dict(async_enabled=enabled,kv_bytes=expected,flag=flag,candidate_mounts=8))
print(json.dumps(dict(passed=True,rank=rank,scope='candidate dry-run; no deployment',modes=results)))
