"""Tokenize archived requests without inference; report counts and hashes only."""
import json,subprocess,urllib.request,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;key=Path('/home/jon/.config/qwen38/api-key').read_text().strip();root='/home/jon/artifact-agent-orchestration/.artifact-agent/baseline-comparison/20260914-temperature-until-1000/batch-001';results=[]
for arm,n,expected in [('temperature-0p4-r01',52,(57216,45397)),('temperature-0p6-r01',48,(69717,49633)),('temperature-0p8-r01',43,(65080,47505))]:
 sequences=[]
 for index in (n-1,n):
  raw=subprocess.check_output(['ssh','jon@192.168.0.167','cat',f'{root}/{arm}/model-input-{index:03}.json'],text=True,timeout=20);data=json.loads(raw)
  body={k:data[k] for k in ('model','messages','tools','chat_template_kwargs') if k in data};body['add_generation_prompt']=True
  req=urllib.request.Request('http://127.0.0.1:30001/tokenize',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=60) as r:answer=json.load(r)
  sequences.append(answer['tokens'])
 a,b=sequences;common=0
 for x,y in zip(a,b):
  if x!=y:break
  common+=1
 row={'arm':arm,'request':n,'before_tokens':len(a),'after_tokens':len(b),'expected_usage':expected,'tokenization_matches_usage':(len(a),len(b))==expected,'common_prefix_tokens':common,'common_prefix_fraction_of_new_prompt':common/len(b),'before_sha256':hashlib.sha256(json.dumps(a).encode()).hexdigest(),'after_sha256':hashlib.sha256(json.dumps(b).encode()).hexdigest()};results.append(row);(p/'cold-prefix-token-check.json').write_text(json.dumps(results,indent=2));print(json.dumps(row),flush=True)
 assert row['tokenization_matches_usage'],'Tokenizer route differs from inference prompt, do not claim exact prefix coverage'
