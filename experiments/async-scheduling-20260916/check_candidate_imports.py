"""Import the integrated candidate in an isolated process, never the engine."""
import importlib.util
import json
from pathlib import Path
import sys

P = Path(sys.argv[1])


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, P/file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


completion = load('vllm.distributed.kv_transfer.kv_connector.v1.gb10_completion', 'completion.py')
connector = load('vllm.distributed.kv_transfer.kv_connector.v1.gb10_aligned_offloading_connector', 'connector.py')
parking = load('vllm.v1.core.sched.gb10_parking_scheduler', 'parking_scheduler.py')
load('vllm.v1.kv_offload.gb10_terminal_state', 'terminal_state.py')
load('vllm.v1.kv_offload.gb10_terminal_versions', 'terminal_versions.py')
load('vllm.v1.kv_offload.gb10_terminal_worker', 'terminal_worker.py')
load('vllm.v1.kv_offload.gb10_gpu_completion_copy', 'gpu_completion_copy.py')
load('vllm.v1.worker.gpu.model_runner', 'model_runner.py')
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
assert issubclass(parking.GB10AsyncParkingScheduler, AsyncScheduler)
assert not issubclass(parking.GB10ParkingScheduler, AsyncScheduler)
assert parking.GB10AsyncParkingScheduler._update_after_schedule is AsyncScheduler._update_after_schedule
assert parking.GB10AsyncParkingScheduler._update_request_with_output is AsyncScheduler._update_request_with_output
assert parking.GB10ParkingScheduler._update_after_schedule is Scheduler._update_after_schedule
# Legacy positional metadata creation must retain its old meaning.
meta = completion.CompletionMetadata({}, {}, {10}, {'R': 'save'})
assert meta.jobs_to_flush == {10} and meta.completion_saves == {'R': 'save'}
assert not meta.terminal_versions and not meta.terminal_releases
save = completion.CompletionSave('record', (), 4)
assert save.terminal_version is None
con = object.__new__(connector.GB10AlignedOffloadingConnector)
con._terminal_versions = {}
con._terminal_releases = []
con._terminal_generation = 0
con._terminal_capacity = 1
assert con.reserve_terminal('A') and not con.reserve_terminal('B')
old = con.terminal_version('A')
con.release_terminal('A')
assert con.reserve_terminal('A') and con.terminal_version('A') > old
assert con._terminal_releases == [('A', old)]
print(json.dumps(dict(passed=True, checks=[
    'integrated modules import', 'upstream async accounting preserved',
    'synchronous MRO preserved', 'legacy metadata positional fields preserved',
    'terminal capacity and generation release',
])))
