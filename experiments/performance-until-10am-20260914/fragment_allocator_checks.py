"""Actual live-source CPU allocator ownership fixture; no GPU execution."""
import ast,json
from types import SimpleNamespace as N
from pathlib import Path
from vllm.v1.kv_offload.rank_local_disk import DiskSlotManager
from vllm.v1.kv_offload.base import ReqContext,LookupResult
source=Path('/tmp/fragment-cleanup-candidate.py').read_text();tree=ast.parse(source)
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CompletionCache')
fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_discard_unreferenced_pages')
env={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'candidate_helper','exec'),env)
ctx=ReqContext('fragment-fixture');m=DiskSlotManager(16)
keys=[b'missing',b'orphan',b'pinned',b'live',b'loading',b'ready',b'pending',b'inherited']
r=m.prepare_store(keys,ctx);assert r is not None;m.complete_store(keys,ctx)
def record(*keys):return N(pages=[(0,i,k) for i,k in enumerate(keys)])
m.discard_idle_keys({b'missing'});m.prepare_load([b'pinned'],ctx)
cache=N(scheduler=N(manager=m),records={0:record(b'live')},selected={0:record(b'loading')},ready={0:record(b'ready')},pending={0:(N(record=record(b'pending')),None,None)},restored_pages={0:{(4,0):b'inherited'}})
retired=record(*keys);before=m._get_num_free_blocks();env[fn.name](cache,[retired])
assert m._get_num_free_blocks()==before+1
assert m.lookup(b'orphan',ctx)==LookupResult.MISS
assert all(m.lookup(k,ctx)==LookupResult.HIT for k in keys[2:])
m.complete_load([b'pinned'],ctx);env[fn.name](cache,[retired]);assert m.lookup(b'pinned',ctx)==LookupResult.MISS
assert all(m.lookup(k,ctx)==LookupResult.HIT for k in keys[3:])
assert m._num_evictable_cache_blocks==5
print(json.dumps({'passed':True,'orphan_reclaimed':True,'pinned_fragment_preserved_until_release':True,'all_five_ownership_sources_preserved':True,'scope':'Actual CPU DiskSlotManager with candidate helper; live model and lookup integration not exercised.'},indent=2))
