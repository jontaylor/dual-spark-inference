import pathlib,subprocess,shlex,json,tarfile,hashlib
P=pathlib.Path(__file__).resolve().parent;REMOTE='jon@192.168.0.167';base='/home/jon/artifact-agent-orchestration/.artifact-agent/baseline-comparison';control='/home/jon/artifact-agent-orchestration/.artifact-agent/restart-reset-repeat/status.json'
def remote(code,*args):return subprocess.check_output(['ssh',REMOTE,shlex.join(['python3','-c',code,*args])],text=True,timeout=120)
check='''import pathlib,json,sys,hashlib
p=pathlib.Path(sys.argv[1]);d=json.loads(pathlib.Path(sys.argv[2]).read_text());assert p.name in ('20260912T023454Z-overnight-baseline-load','20260912T122137Z-staggered-baseline-load');assert json.loads((p/'campaign.json').read_text())['status']=='finished';assert str(p)!=d['parent'] and str(p)!=d['current'];assert p.parent.name=='baseline-comparison'
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 if proc.name==str(__import__('os').getpid()):continue
 try:c=(proc/'cmdline').read_bytes()
 except (OSError,PermissionError):continue
 assert not any(x.startswith(str(p).encode()) for x in c.split(bytes([0]))),'live process reference'
files={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and not f.is_symlink()};print(json.dumps(files))
'''
for name in __import__('sys').argv[1:]:
 path=base+'/'+name;manifest=json.loads(remote(check,path,control));dest=P/(name+'.full.tar.gz')
 with dest.open('wb') as f:subprocess.run(['ssh',REMOTE,shlex.join(['tar','-czf','-','-C',base,name])],stdout=f,check=True,timeout=180)
 with tarfile.open(dest) as t:
  seen=set()
  for member in t:
   rel=member.name.removeprefix(name+'/')
   if rel in manifest:
    assert hashlib.sha256(t.extractfile(member).read()).hexdigest()==manifest[rel],(name,rel);seen.add(rel)
  assert seen==set(manifest)
 assert json.loads(remote(check,path,control))==manifest,'archive source changed'
 (P/(name+'.archive-manifest.json')).write_text(json.dumps({'source':path,'files':manifest,'archive_sha256':hashlib.sha256(dest.read_bytes()).hexdigest()},indent=2))
 code='''import pathlib,json,sys,shutil,hashlib
p=pathlib.Path(sys.argv[1]);expected=json.load(sys.stdin);assert p.parent==pathlib.Path(sys.argv[2]) and p.name in ('20260912T023454Z-overnight-baseline-load','20260912T122137Z-staggered-baseline-load') and not p.is_symlink();actual={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and not f.is_symlink()};assert actual==expected;shutil.rmtree(p)
'''
 subprocess.run(['ssh',REMOTE,shlex.join(['python3','-c',code,path,base])],input=json.dumps(manifest),text=True,check=True,timeout=120)
 print('Verified full archive and removed inactive original:',name,flush=True)
