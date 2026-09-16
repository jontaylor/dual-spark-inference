"""Resume only this transition's recorded PIDs after complete H live gates."""
import json,subprocess,shlex,time
from pathlib import Path
p=Path(__file__).resolve().parent
assert json.loads((p/'validation-H/summary.json').read_text())['passed']
s=json.loads((p/'disk-oracle-H/summary.json').read_text())
assert s['disk_load_bytes']>0 and s['independent_load_bytes']==0
assert s['disk_cached']==s['independent_cached']==s['boundary']
assert s['exact_prefix_matches'] and s['tokens_equal'] and s['scores_equal']
inherited=json.loads((p/'inherited-model-H/summary.json').read_text())
assert any(r['inherited_pages']>0 and r['tokens_equal'] and r['scores_equal'] and r['cached']==7360 for r in inherited)
pause=json.loads((p/'next-candidate-pause.json').read_text())
code='''import json,os,signal,time
from pathlib import Path
pause=PAUSE
results={}
for pid,info in pause['processes'].items():
 p=Path('/proc')/pid/'stat'
 if not p.exists():results[pid]='already_exited';continue
 f=p.read_text().rsplit(') ',1)[1].split();assert f[19]==info['starttime'], 'PID reused'
 if f[0]=='Z':results[pid]='already_exited';continue
 os.kill(int(pid),signal.SIGCONT);results[pid]='resumed'
out={'time':time.time(),'processes':results,'configuration':'H','scope':'Final1.2arm spans C/H; seven completed arms were C-only'}
(Path(pause['record'])/'server-transition-H-resume.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
'''.replace('PAUSE',repr(pause))
r=subprocess.check_output(['ssh','jon@192.168.0.167','python3 -'],input=code,text=True,timeout=30);(p/'H-workload-resume.json').write_text(r);print(r,flush=True)
# Watchdog owns its controller pause and confirms its own identity-checked resume.
(p/'fine-boundary-release').touch()
msg='H live validation and independent disk-vs-resident oracle passed. Recorded remaining1.2arm/supervisor resumed; controller-hold release requested via watchdog (separate confirmation follows). H uses inherited full-page storage keys, measured fixed-K fine decode GEMMs, MTP3 and FULL transfer verification. Seven completed batch001arms are C-only; last1.2arm spans C/H. Continue original frozen workload, sampling and retries; first full next cycle will start on H.'
subprocess.run(['ssh','jon@192.168.0.167',shlex.join(['codex','queue','--thread','01a09b9f-c6c7-7bb0-8c25-b22748f43388','--message',msg])],check=True,timeout=30)
