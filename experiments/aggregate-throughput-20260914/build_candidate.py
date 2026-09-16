from pathlib import Path
import json,hashlib,ast,types,difflib
p=Path(__file__).resolve().parent;root=p.parents[1];base=root/'experiments/targeted-determinism-20260913/completion.fixed.py';s=base.read_text();needle='            or boundary <= 0\n';assert s.count(needle)==1
s=s.replace(needle,needle+'            # Lookup cannot reuse an unaligned completion. Do not retain or\n            # spill its pages. Active pressure snapshots use a separate path.\n            or (not active and boundary % 64 != 0)\n')
f=p/'completion.no_waste.py';f.write_text(s);compile(s,str(f),'exec')
(p/'completion.no_waste.patch').write_text(''.join(difflib.unified_diff(base.read_text().splitlines(True),s.splitlines(True),fromfile=str(base),tofile=str(f))))
tree=ast.parse(s);fn=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='finish')
ns={'eligible':lambda r:True,'RequestStatus':types.SimpleNamespace(FINISHED_STOPPED='stop',FINISHED_LENGTH_CAPPED='length')};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(f),'exec'),ns)
class ReachedAllocation(Exception):pass
class State:
 def update_offload_keys(self):raise ReachedAllocation
obj=types.SimpleNamespace(scheduler=types.SimpleNamespace(_req_status={'r':State()}),selected={},connector=types.SimpleNamespace(_alignment=1600),pending={},ready={})
checks=[]
for boundary,active,expect_skip in [(193,False,True),(192,False,False),(193,True,False),(1601,False,True),(1600,False,False)]:
 r=types.SimpleNamespace(request_id='r',num_computed_tokens=boundary,num_tokens=boundary+1,num_in_flight_tokens=0,status='stop')
 try:
  value=ns['finish'](obj,r,None,active=active,memory=True);assert expect_skip and value is False;result='skipped_before_allocation'
 except ReachedAllocation:assert not expect_skip;result='capture_path_retained'
 checks.append({'boundary':boundary,'active':active,'result':result})
(p/'capture-eligibility-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
for rank in [0,1]:
 cfg=json.loads((root/f'experiments/spec-determinism-20260914/candidate-r{rank}.json').read_text());key='distributed/kv_transfer/kv_connector/v1/gb10_completion.py';cfg['runtime_overrides'][key]=str(f);cfg['runtime_override_sha256'][key]=hashlib.sha256(f.read_bytes()).hexdigest();(p/f'candidate-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
print('Eligibility tests pass; candidate removes only completion snapshots that lookup already rejects.')
