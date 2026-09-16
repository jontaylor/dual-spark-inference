import hashlib,json,pathlib,sys
root=pathlib.Path('/home/jon/dual-spark-inference-kv-paging');exp=root/'experiments/fla-threshold-20260912'
p=root/'deploy_config.json';c=json.loads(p.read_text());rel='third_party/flash_linear_attention/ops/utils.py'
if sys.argv[1]=='patched':
 (exp/'config-before.json').write_text(p.read_text())
 assert rel not in c.get('runtime_overrides',{})
 source=exp/'utils.patched.py'
 c.setdefault('runtime_overrides',{})[rel]=str(source)
 c.setdefault('runtime_override_sha256',{})[rel]=hashlib.sha256(source.read_bytes()).hexdigest()
else:
 assert c.get('runtime_overrides',{}).get(rel)==str(exp/'utils.patched.py')
 for key in ('runtime_overrides','runtime_override_sha256'):
  c[key].pop(rel)
  if not c[key]:del c[key]
p.write_text(json.dumps(c,indent=2)+'\n')
print(sys.argv[1])
