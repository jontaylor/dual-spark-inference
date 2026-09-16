"""Reproduce the deployed same-pass park/restore crash and verify its fix."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace as N

P = Path(__file__).parent


def load(path):
    tree = ast.parse(path.read_text())
    functions = [f for cls in tree.body if isinstance(cls, ast.ClassDef)
                 for f in cls.body if isinstance(f, ast.FunctionDef)
                 and f.name == 'build_connector_meta']
    assert len(functions) == 1
    ns = dict(SchedulerOutput=object, KVConnectorMetadata=object,
              ScheduleEndContext=N, OffloadingConnectorMetadata=N)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), ns)
    return ns['build_connector_meta']


def fixture(store=False, current=True, owner='A'):
    jobs = {2: N(req_id=owner, is_store=False)}
    if store:
        jobs[1] = N(req_id='A', is_store=True)
    return N(_update_req_states=lambda output: None,
             manager=N(on_schedule_end=lambda ctx: None),
             _req_status={'A': N(transfer_jobs=set(jobs))}, _jobs=jobs,
             _current_batch_load_jobs={2: N(req_id=owner)} if current else {},
             _current_batch_jobs_to_flush=set(), _block_id_to_pending_jobs={},
             _current_batch_allocated_block_ids=set(), automatic_stores_enabled=False)


output = N(scheduled_new_reqs=[], preempted_req_ids={'A'}, finished_req_ids=set())
original = load(P.parent/'allocation-deferral-implementation-20260914/offloading_scheduler.py')
fixed = load(P/'offloading_scheduler.py')
try:
    original(fixture(), output)
except AssertionError:
    pass
else:
    raise AssertionError('Original crash was not reproduced')
for store in (False, True):
    state = fixture(store=store)
    meta = fixed(state, output)
    assert set(meta.load_jobs) == {2}
    assert meta.jobs_to_flush == ({1} if store else set())
    assert state._req_status['A'].transfer_jobs == ({1, 2} if store else {2})
    assert state._current_batch_load_jobs == {}
for state in (fixture(current=False), fixture(owner='another-request')):
    try:
        fixed(state, output)
    except AssertionError:
        pass
    else:
        raise AssertionError('Invalid preemption silently accepted')
print(json.dumps(dict(passed=True, checks=[
    'original load-as-store assertion reproduced',
    'same-pass restore retained and not flushed before submission',
    'old stores still flushed', 'old or unrelated loads still rejected',
])))
