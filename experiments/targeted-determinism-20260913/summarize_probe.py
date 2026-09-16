import json,sys
from pathlib import Path
root=Path(__file__).resolve().parent
out=[]
for name in sys.argv[1:]:
 rows=json.loads((root/name/'results.json').read_text());ref=rows[0]
 for row in rows:
  tokens=row['token_ids'];rt=ref['token_ids'];first=next((i for i,(a,b) in enumerate(zip(rt,tokens)) if a!=b),None)
  if first is None and len(rt)!=len(tokens):first=min(len(rt),len(tokens))
  out.append({'experiment':name,'name':row['name'],'error':row['error'],'tokens':len(tokens),'seconds':row['seconds'],'canonical_equal':row['canonical']==ref['canonical'],'prompt_equal':row['prompt_token_ids']==ref['prompt_token_ids'],'tokens_equal':tokens==rt,'scores_equal':row['logprobs']==ref['logprobs'],'first_token_difference':first,'usage':row['usage']})
print(json.dumps(out,indent=2));(root/'probe-summary.json').write_text(json.dumps(out,indent=2))
