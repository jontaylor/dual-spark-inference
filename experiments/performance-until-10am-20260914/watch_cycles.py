import datetime,json,os,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;end=datetime.datetime(2026,9,14,9,tzinfo=datetime.timezone.utc).timestamp()
(p/'cycle-watch-process.json').write_text(json.dumps({'pid':os.getpid(),'starttime':Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ',1)[1].split()[19],'deadline':end}))
code='''import json,time
from pathlib import Path
base=Path('/home/jon/artifact-agent-orchestration/.artifact-agent/baseline-comparison/20260914-temperature-until-1000');s=json.loads((base/'status.json').read_text());pid=s['pid'];proc=Path('/proc')/str(pid)/'stat';fields=proc.read_text().rsplit(') ',1)[1].split() if proc.exists() else []
out={'time':time.time(),'controller':s,'controller_process_state':fields[0] if fields else 'gone','controller_process_starttime':fields[19] if fields else None,'cycles':[]}
for record in s['batches']:
 p=Path(record);t=p/'telemetry.json'
 if not t.exists():continue
 d=json.loads(t.read_text());rows=[{k:r.get(k) for k in ['id','status','generated_tokens','max_prompt_tokens','verification']} for r in d['rows']];out['cycles'].append({'record':record,'rows':rows})
print(json.dumps(out))
'''
with (p/'cycles.jsonl').open('a') as f:
 while time.time()<end:
  row={'time':time.time()}
  try:
   result=subprocess.run(['ssh','jon@192.168.0.167','python3 -'],input=code,text=True,capture_output=True,check=True,timeout=20);row['workload']=json.loads(result.stdout)
  except Exception as e:row['error']=repr(e)
  f.write(json.dumps(row)+'\n');f.flush()
  if 'workload' in row:
   w=row['workload'];last=w['cycles'][-1] if w['cycles'] else {};print(json.dumps({'time':row['time'],'cycle':w['controller'].get('current'),'process_state':w['controller_process_state'],'arms':{r['id']:r['status'] for r in last.get('rows',[])}}),flush=True)
  time.sleep(min(60,max(0,end-time.time())))
print('Cycle monitoring deadline reached',flush=True)
