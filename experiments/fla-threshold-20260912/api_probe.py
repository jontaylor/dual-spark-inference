"""Cold-prefix retrieval/arithmetic probes under unchanged background load."""
import datetime,json,pathlib,sys,time,urllib.request,uuid
root=pathlib.Path(__file__).resolve().parent
key=pathlib.Path('/home/jon/.config/qwen38/api-key').read_text().strip()
base='http://127.0.0.1:30001'
def post(path,payload):
 return urllib.request.urlopen(urllib.request.Request(base+path,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+key}),timeout=600)
results=[]
for rep in range(3):
 for target in [2048,8192]:
  marker='FALCON-'+str(713+rep)
  prefix='Independent test '+uuid.uuid4().hex+'\n'
  line='Archive entry: this ordinary record contains no requested access code.\n'
  tail=f'\nReturn exactly the access code, then a space, then the sum of 137 and 286. No explanation.'
  n=target//13
  for _ in range(4):
   text=prefix+line*(n//2)+f'\nThe access code is {marker}.\n'+line*(n-n//2)+tail
   messages=[{'role':'user','content':text}]
   common={'model':'qwen3.8-flash-next','messages':messages,'chat_template_kwargs':{'enable_thinking':False}}
   with post('/tokenize',common) as r:d=json.load(r)
   count=d.get('count',len(d.get('tokens',[])))
   if abs(count-target)<20:break
   n=max(1,n+round((target-count)/13))
  started=datetime.datetime.now(datetime.timezone.utc).isoformat();t=time.monotonic();first=None;out='';usage=None;metrics=None;finish=None
  with post('/v1/chat/completions',{**common,'temperature':0,'max_tokens':48,'stream':True,'stream_options':{'include_usage':True}}) as r:
   for raw in r:
    s=raw.decode().strip()
    if not s.startswith('data: ') or s=='data: [DONE]':continue
    d=json.loads(s[6:]);usage=d.get('usage') or usage;metrics=d.get('metrics') or metrics
    for c in d.get('choices',[]):
     piece=c.get('delta',{}).get('content') or ''
     if piece and first is None:first=time.monotonic()-t
     out+=piece;finish=c.get('finish_reason') or finish
  row={'arm':sys.argv[1],'utc':started,'rep':rep,'target_tokens':target,'prompt_tokens':count,'ttft_s':first,'elapsed_s':time.monotonic()-t,'text':out,'correct':out.strip()==marker+' 423','usage':usage,'metrics':metrics,'finish':finish}
  results.append(row);print(json.dumps(row),flush=True)
  (root/(sys.argv[1]+'.json')).write_text(json.dumps(results,indent=2))
