import subprocess,json,datetime,re,shlex
from pathlib import Path
out=Path(__file__).parent
secret=Path('/home/jon/.config/qwen38/api-key').read_text().strip()
def clean(v):
 if isinstance(v,str):
  if secret:v=v.replace(secret,'<REDACTED_API_KEY>')
  if '=' in v:
   k=v.split('=',1)[0]
   if re.search(r'(PASSWORD|SECRET|TOKEN|API_KEY|ACCESS_KEY|PRIVATE_KEY)',k,re.I):return k+'=<REDACTED_SECRET>'
  return v
 if isinstance(v,list):return [clean(x) for x in v]
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 return v
script=r'''import json,subprocess,hashlib
from pathlib import Path
name=NAME
info=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
cmd=['docker','exec',name,'python3','-c',"import pathlib,json; p=pathlib.Path('/proc'); o=[];\nfor d in p.iterdir():\n if d.name.isdigit():\n  try:\n   a=(d/'cmdline').read_bytes().split(b'\\0'); e=(d/'environ').read_bytes().split(b'\\0'); o.append({'pid':int(d.name),'argv':[x.decode(errors='replace') for x in a if x],'environment':[x.decode(errors='replace') for x in e if x]})\n  except (OSError,PermissionError):pass\nprint(json.dumps(o))"]
processes=json.loads(subprocess.check_output(cmd))
mounts=[m for m in info['Mounts'] if m['Destination'].endswith('.py')]
hashes={}
for m in mounts:
 v=subprocess.check_output(['docker','exec',name,'sha256sum',m['Destination']],text=True).split()[0]
 hashes[m['Destination']]={'sha256':v,'source':m['Source']}
print(json.dumps({'container':info,'processes':processes,'mounted_python_hashes':hashes}))
'''
for rank in [0,1]:
 s=script.replace('NAME',repr('qwen38-kv-paging-r'+str(rank)))
 cmd=['python3','-'] if rank==0 else ['ssh','jon@192.168.100.11','python3 -']
 p=subprocess.run(cmd,input=s,text=True,capture_output=True,check=True,timeout=60)
 data=clean(json.loads(p.stdout)); data['captured_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (out/f'live-r{rank}.json').write_text(json.dumps(data,indent=2))
 print(rank,data['container']['State']['Status'],data['container']['State']['StartedAt'],'python_mounts',len(data['mounted_python_hashes']))
