import pathlib,json,urllib.request,time,threading,concurrent.futures,uuid
p=pathlib.Path(pathlib.Path('/tmp/prefill1024-path').read_text());key=pathlib.Path('/home/jon/.config/qwen38/api-key').read_text().strip();barrier=threading.Barrier(16);tag=uuid.uuid4().hex
start=time.time()
def run(i):
 body={'model':'qwen3.8-flash-next','prompt':f'{tag} request {i}: '+ ' blue'*1850+'\nWrite a short sentence.','max_tokens':16,'temperature':0,'stream':True,'stream_options':{'include_usage':True}}
 req=urllib.request.Request('http://127.0.0.1:30001/v1/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 barrier.wait();t=time.time();first=None;usage=None;rid=None
 with urllib.request.urlopen(req,timeout=180) as r:
  for raw in r:
   line=raw.decode().strip()
   if not line.startswith('data: ') or line=='data: [DONE]':continue
   d=json.loads(line[6:]);rid=d.get('id',rid)
   if first is None and any(c.get('text') for c in d.get('choices',[])):first=time.time()
   if d.get('usage'):usage=d['usage']
 return {'i':i,'id':rid,'sent':t,'first':first,'ttft':first-t if first else None,'finished':time.time(),'usage':usage}
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:results=list(ex.map(run,range(16)))
(p/'probe.json').write_text(json.dumps({'start':start,'results':results},indent=2));print(json.dumps({'requests':len(results),'ttft_sorted':[round(r['ttft'],2) for r in sorted(results,key=lambda x:x['ttft'])],'usage':results[0]['usage']}))
