import pathlib,subprocess,time,json
P=pathlib.Path(__file__).resolve().parent
pid=int(__import__('sys').argv[1])
start=pathlib.Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]
procs=[]
try:
 for rank in (0,1):
  cmd=['nvidia-smi','--query-gpu=timestamp,utilization.gpu,utilization.memory,power.draw,clocks.sm,temperature.gpu','--format=csv,noheader,nounits','-lms','500']
  if rank:cmd=['ssh','-o','ConnectTimeout=10','jon@192.168.100.11',*cmd]
  f=(P/f'gpu-continuous-r{rank}.csv').open('a');proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);procs.append((proc,f))
 (P/'gpu-watch.json').write_text(json.dumps({'controller_pid':pid,'controller_start_ticks':start,'samplers':[p.pid for p,f in procs],'started':time.time()}))
 while True:
  try:s=pathlib.Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
  except FileNotFoundError:break
  if s[19]!=start or s[0]=='Z':break
  if any(p.poll() is not None for p,f in procs):raise RuntimeError('GPU sampler exited')
  time.sleep(5)
finally:
 for p,f in procs:
  p.terminate()
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:p.kill();p.wait()
  f.close()
