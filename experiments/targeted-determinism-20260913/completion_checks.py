import ast, json, types
from pathlib import Path
from collections import OrderedDict
from array import array
import hashlib
p=Path('/e/completion.fixed.py')
tree=ast.parse(p.read_text())
lookup=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='lookup')
funs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('prefix_digests','eligible')]
ns=dict(hashlib=hashlib,array=array,LookupResult=types.SimpleNamespace(MISS='miss',HIT_PENDING='pending'))
exec(compile(ast.Module(body=funs+[lookup],type_ignores=[]),str(p),'exec'),ns)
req=types.SimpleNamespace(request_id='r',all_token_ids=list(range(300)),cache_salt=None,num_prompt_tokens=300,mm_features=[],prompt_embeds=None,lora_request=None,skip_reading_prefix_cache=False)
state=types.SimpleNamespace(transfer_jobs=[],req_context=None)
s=types.SimpleNamespace(_req_status={'r':state},manager=types.SimpleNamespace(lookup=lambda *args:'hit'))
obj=types.SimpleNamespace(selected={},scheduler=s,pending={},active=set(),cancelled=set(),records=OrderedDict())
for boundary in (128,193):
    digest=ns['prefix_digests'](req.all_token_ids,None,[boundary+1])[boundary+1]
    obj.records[(boundary+1,digest)]=types.SimpleNamespace(witness_length=boundary+1,digest=digest,boundary=boundary,pages=[(0,0,b'key')])
assert ns['lookup'](obj,req,0)==(128,True)
assert obj.selected['r'].boundary==128
assert ns['lookup'](obj,req,128) is None
record=next(r for r in obj.records.values() if r.boundary==193)
obj.records.clear();obj.pending={'p':(types.SimpleNamespace(record=record),None,None)}
assert ns['lookup'](obj,req,0) is None
out={'aligned_snapshot_restored':True,'unaligned_snapshot_skipped':True,'unaligned_pending_snapshot_does_not_block':True}
Path('/e/completion-checks.json').write_text(json.dumps(out,indent=2));print(out)
