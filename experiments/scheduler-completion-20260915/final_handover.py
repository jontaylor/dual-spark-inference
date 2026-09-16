"""Wait for existing arm35 controller to exit, then launch final arm36 with target16."""
import json,pathlib,subprocess,time
P=pathlib.Path(__file__).resolve().parent
old=json.loads((P/'launch.json').read_text())
def alive(pid,start=None):
 try:s=pathlib.Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
 except FileNotFoundError:return False
 return s[0]!='Z' and (start is None or s[19]==start)
while alive(old['pid'],old['start_ticks']):time.sleep(5)
status=json.loads((P/'status.json').read_text())
assert status['phase']=='all_requested_arms_complete',status
out=P/'35-s32-b4096-t1024'
done=json.loads((out/'complete.json').read_text());assert done['completed']>=8 and done['same_campaign']
for key in ('gpu_watch.py_pid','report_watch.py_pid'):
 while alive(old[key]):time.sleep(2)
# Gate remains until the old controller has definitely exited.
gate=P/'36-deferred-controller-handover'
assert json.loads((gate/'skipped.json').read_text())['temporary']
(gate/'skipped.json').unlink();gate.rmdir()
assert not (P/'36-s32-b16384-t1024').exists()
(P/'launch-before-final-handover.json').write_text(json.dumps(old,indent=2)+'\n')
with (P/'controller.log').open('a') as f:
 proc=subprocess.Popen(['python3','-u',str(P/'run.py'),'36'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
identity={'pid':proc.pid,'start_ticks':pathlib.Path(f'/proc/{proc.pid}/stat').read_text().split(') ',1)[1].split()[19],'launched':time.time(),'resume_arm':36,'reason':'User requested final s32/b16384/t1024 run to all16 task completions'}
(P/'launch.json').write_text(json.dumps(identity,indent=2)+'\n')
for script in ('gpu_watch.py','report_watch.py'):
 with (P/(script+'.log')).open('a') as f:
  args=['python3','-u',str(P/script)]+([str(proc.pid)] if script=='gpu_watch.py' else [])
  watcher=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  identity[script+'_pid']=watcher.pid
(P/'launch.json').write_text(json.dumps(identity,indent=2)+'\n')
print(json.dumps(identity),flush=True)
