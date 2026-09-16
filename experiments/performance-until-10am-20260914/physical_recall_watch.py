"""Observe immutable blob generations via metadata, without reading cache contents."""
import ctypes,datetime,json,os,select,signal,struct,subprocess,sys,time
from pathlib import Path
rank=int(sys.argv[1]);p=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else Path(__file__).resolve().parent;p.mkdir(parents=True,exist_ok=True)
root=Path('/home/jon/.cache/vllm-gb10-kv-paging');matches=list(root.glob(f'*/rank-{rank}/content'));assert len(matches)==1,matches
folder=matches[0];options=subprocess.check_output(['findmnt','-n','-o','OPTIONS','-T',str(folder)],text=True).strip();assert 'noatime' not in options and ('relatime' in options or 'strictatime' in options),options
lib=ctypes.CDLL(None,use_errno=True);fd=lib.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC);assert fd>=0
mask=0x100|0x80|0x8|0x200|0x400|0x800 # CREATE,MOVED_TO,CLOSE_WRITE,DELETE,DELETE_SELF,MOVE_SELF
wd=lib.inotify_add_watch(fd,os.fsencode(folder),mask);assert wd>=0
end=datetime.datetime(2026,9,14,9,tzinfo=datetime.timezone.utc).timestamp();active={};retired=[];errors=[];start=time.time();running=True
status_path=p/f'physical-recall-r{rank}.json';events=(p/f'physical-recall-r{rank}.jsonl').open('a');process={'pid':os.getpid(),'starttime':Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ',1)[1].split()[19],'rank':rank,'folder':str(folder),'started':start,'mount_options':options};(p/f'physical-recall-process-r{rank}.json').write_text(json.dumps(process,indent=2))
def stop(*args):
 global running
 running=False
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def describe(entry):
 st=os.fstat(entry['fd'])
 return {'name':entry['name'],'preexisting':entry['preexisting'],'attached':entry['attached'],'size':st.st_size,'atime_ns':st.st_atime_ns,'mtime_ns':st.st_mtime_ns,'observed_read':st.st_atime_ns>st.st_mtime_ns}
def attach(name,preexisting=False):
 if not name.endswith('.bin') or name in active:return
 try:f=os.open(folder/name,os.O_PATH|os.O_CLOEXEC)
 except FileNotFoundError:return
 active[name]={'fd':f,'name':name,'preexisting':preexisting,'attached':time.time()}
def retire(name):
 e=active.pop(name,None)
 if e is None:return
 row=describe(e);os.close(e['fd']);row.update(event='unlinked',time=time.time());retired.append(row);events.write(json.dumps(row)+'\n');events.flush()
def summary():
 live=[describe(x) for x in active.values()];cohorts={}
 for label,rows in [('new_retired',[r for r in retired if not r['preexisting']]),('new_live',[r for r in live if not r['preexisting']]),('preexisting_retired',[r for r in retired if r['preexisting']]),('preexisting_live',[r for r in live if r['preexisting']])]:
  cohorts[label]={'generations':len(rows),'bytes':sum(r['size'] for r in rows),'read_generations':sum(r['observed_read'] for r in rows),'read_bytes':sum(r['size'] for r in rows if r['observed_read']),'unread_bytes':sum(r['size'] for r in rows if not r['observed_read'])}
 result={'time':time.time(),'started':start,'rank':rank,'folder':str(folder),'running':running,'errors':errors,'cohorts':cohorts,'scope':'Immutable physical blob generations. Metadata only; relatime updates first read after write, including reads through slot hardlinks. Preexisting blobs are separate. Unlinks during server cleanup must be excluded from eviction conclusions; no-reader attribution by PID. O_PATH descriptors closed immediately on observed unlink.'};status_path.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
for file in folder.iterdir():attach(file.name,True)
last=0
try:
 while running and time.time()<end:
  if time.time()-last>=60:summary();last=time.time()
  readable,_,_=select.select([fd],[],[],1)
  if not readable:continue
  data=os.read(fd,1<<20);pos=0
  while pos<len(data):
   watch,bits,cookie,length=struct.unpack_from('iIII',data,pos);pos+=16;name=data[pos:pos+length].split(b'\0',1)[0].decode();pos+=length
   if bits&0x4000:errors.append({'time':time.time(),'error':'inotify_queue_overflow'});running=False;break
   if bits&0x200:retire(name)
   if bits&(0x100|0x80|0x8):attach(name)
   if bits&(0x400|0x800):errors.append({'time':time.time(),'event':'directory_ended'});running=False
finally:
 running=False;summary()
 for e in active.values():os.close(e['fd'])
 events.close();os.close(fd)
