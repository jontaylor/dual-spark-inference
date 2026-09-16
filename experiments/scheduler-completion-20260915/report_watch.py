import json,pathlib,subprocess,time
P=pathlib.Path(__file__).resolve().parent
identity=json.loads((P/'launch.json').read_text());last=None
while True:
 try:
  stat=pathlib.Path(f"/proc/{identity['pid']}/stat").read_text().split(') ',1)[1].split()
  alive=stat[19]==identity['start_ticks'] and stat[0]!='Z'
 except FileNotFoundError:alive=False
 current=tuple(sorted(str(p) for p in P.glob('[0-9][0-9]-*/complete.json')))
 if current!=last or not alive:
  for script in ('analyze.py','audit.py'):
   subprocess.run(['python3',str(P/script)],check=True,stdout=subprocess.DEVNULL)
  last=current
  (P/'report-watch-status.json').write_text(json.dumps({'time':time.time(),'controller_alive':alive,'completed_arms':len(current)},indent=2))
 if not alive:break
 time.sleep(15)
