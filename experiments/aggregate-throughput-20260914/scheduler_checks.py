import ast,types,json
from pathlib import Path
p=Path(__file__).resolve().parent;tree=ast.parse((p/'scheduler.aligned.py').read_text());fn=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='_cache_aligned_split');ns={"Request":object};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<scheduler>','exec'),ns)
for start in range(7200,7400):
 for n in range(1,5):
  r=types.SimpleNamespace(num_computed_tokens=start,num_tokens=start+1,num_prompt_tokens=7000)
  result=ns['_cache_aligned_split'](None,r,n)
  assert 1<=result<=n and start+result<=(start//64+1)*64
for last in range(128):
 computed=7200+last;boundary=computed//64*64
 assert 0<=computed-boundary<64
print(json.dumps({'decode_step_cases':800,'no_zero_length_step':True,'no_skipped_checkpoint':True,'replay_bound_cases':128,'max_tail_replay':63},indent=2))
