import datetime,hashlib,json,os,time,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent; deadline=datetime.datetime(2026,9,14,9,tzinfo=datetime.timezone.utc).timestamp()
(p/'monitor-process.json').write_text(json.dumps({'pid':os.getpid(),'starttime':Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ',1)[1].split()[19],'deadline_utc':deadline,'started':time.time()}))
with (p/'metrics.jsonl').open('a') as f:
 while time.time()<deadline:
  row={'time':time.time()}
  try:
   with urllib.request.urlopen('http://127.0.0.1:30001/metrics',timeout=5) as r:raw=r.read().decode()
   row['metrics']={l.rsplit(' ',1)[0]:float(l.rsplit(' ',1)[1]) for l in raw.splitlines() if l.startswith('vllm:') and '_bucket{' not in l and '_created{' not in l and 'config_info{' not in l}
   cfg=p.parents[1]/'deploy_config.json';row['config_sha256']=hashlib.sha256(cfg.read_bytes()).hexdigest()
  except Exception as e:row['error']=repr(e)
  f.write(json.dumps(row)+'\n');f.flush()
  if int(row['time'])%60<10:print(json.dumps({'time':row['time'],'ok':'error' not in row,'config':row.get('config_sha256')}),flush=True)
  time.sleep(10)
print('10AM Europe/London observation deadline reached',flush=True)
