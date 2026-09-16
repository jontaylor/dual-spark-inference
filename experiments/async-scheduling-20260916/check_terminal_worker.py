"""Integrated worker/lifetime test with real Torch CPU storage and Triton interpreter."""
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as N

assert os.environ.get('TRITON_INTERPRET') == '1'
import torch
P = Path(sys.argv[1])
for name, file in (
    ('vllm.distributed.kv_transfer.kv_connector.v1.gb10_completion', 'completion.py'),
    ('vllm.v1.kv_offload.gb10_terminal_state', 'terminal_state.py'),
    ('vllm.v1.kv_offload.gb10_terminal_versions', 'terminal_versions.py'),
    ('vllm.v1.kv_offload.gb10_terminal_worker', 'terminal_worker.py'),
):
    spec = importlib.util.spec_from_file_location(name, P/file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

from vllm.v1.kv_offload.gb10_terminal_worker import TerminalWorker
from vllm.v1.kv_cache_interface import MambaSpec, CircularBufferSpec, KVCacheGroupSpec
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec, get_temporal_copy_spec, is_conv_state_dim_first,
)

axis = 1 if is_conv_state_dim_first() else 0
conv_shape = (4, 6) if axis == 1 else (6, 4)
conv = torch.arange(120, dtype=torch.uint8).reshape(5, *conv_shape)
temporal = torch.arange(80, dtype=torch.uint8).reshape(5, 4, 4)
ring = torch.arange(160, dtype=torch.uint8).reshape(5, 8, 4)
spec = MambaSpec(block_size=16, shapes=(conv_shape, (4, 4)),
                 dtypes=(torch.uint8, torch.uint8))
groups = [KVCacheGroupSpec(layer_names=['m'], kv_cache_spec=spec),
          KVCacheGroupSpec(layer_names=['r'], kv_cache_spec=CircularBufferSpec(
              block_size=8, num_kv_heads=1, head_size=4, dtype=torch.uint8))]
context = {'m': N(kv_cache=(conv, temporal)), 'r': N(kv_cache=ring)}
runner = N(max_num_reqs=2, max_model_len=256, num_speculative_steps=3,
    device=torch.device('cpu'), kv_cache_config=N(num_blocks=5),
    vllm_config=N(compilation_config=N(static_forward_context=context),
                 kv_transfer_config=N(kv_connector_extra_config={'async_terminal_snapshot_bytes':4096})),
    model=N(get_mamba_state_copy_funcs=lambda types: {spec.mamba_type:(get_conv_copy_spec,get_temporal_copy_spec)}),
    block_tables=N(block_tables=[N(gpu=torch.tensor([[1,2,3,4],[1,2,3,4]],dtype=torch.int32)),
                                N(gpu=torch.tensor([[1],[1]],dtype=torch.int32))]),
    req_states=N(req_id_to_index={'A':0}, num_computed_tokens=N(gpu=torch.tensor([100,0],dtype=torch.int32)),
        total_len=N(gpu=torch.tensor([101,0],dtype=torch.int32)),
        prompt_len=N(gpu=torch.tensor([90,0],dtype=torch.int32))),
    model_state=N(num_accepted_tokens_gpu=torch.tensor([4,1],dtype=torch.int32),
                  _mamba_state_idx_gpu=torch.tensor([0,0],dtype=torch.int32)))
connector = N(_completion_groups=groups, _completion_order=(0,1),
              connector_worker=N(worker=N(state_offset=lambda t,g: 24 if t is temporal else 0)))
worker = TerminalWorker(connector, runner)
params = N(stop_token_ids=[], eos_token_id=None, max_tokens=11)


def output(new=(), finished=(), releases=(), versions=None):
    return N(scheduled_new_reqs=[N(req_id=rid,sampling_params=params) for rid in new],
        finished_req_ids=set(finished), preempted_req_ids=set(),
        kv_connector_metadata=N(terminal_versions=versions or {}, terminal_releases=list(releases)))


worker.after_requests(output(new=['A'],versions={'A':1}))
batch = N(num_reqs=1,idx_mapping=torch.tensor([0],dtype=torch.int32))
worker.capture(batch,torch.tensor([[8,8,8,9]],dtype=torch.int64),torch.tensor([4],dtype=torch.int32))
save = N(request_id='A',terminal_version=1,record=N(boundary=100))
payload = worker.selected(save)
assert payload is not None and payload.tag.tolist() == [100,1,1]
saved = payload.data.clone()
expected_conv = torch.zeros_like(conv[1])
if axis == 1:
    expected_conv[:,:3] = conv[1,:,3:]
else:
    expected_conv[:3] = conv[1,3:]
expected_temporal = temporal[4].clone()
expected_ring = ring[1].clone()
assert torch.equal(saved[:24],expected_conv.reshape(-1))

# Simulate queued work overwriting all live source states, followed by a
# finished request losing its worker slot while allocation remains deferred.
conv.zero_(); temporal.zero_(); ring.zero_()
worker.before_requests(output(finished=['A']))
runner.req_states.req_id_to_index = {'B':0}
worker.after_requests(output(new=['B'],versions={'B':2}))
worker.capture(batch,torch.tensor([[8,8,8,9]],dtype=torch.int64),torch.tensor([4],dtype=torch.int32))
assert torch.equal(worker.selected(save).data,saved)
job = N(src_spec=N(group_sizes=[1,1]),dst_spec=N(memory_blocks=[2,2]))
assert worker.resident(save,job)
assert torch.equal(conv[2],expected_conv)
assert torch.equal(temporal[2],expected_temporal)
assert torch.equal(ring[2],expected_ring)
replacements = worker.disk_replacements(save)
assert [offset for offset,_ in replacements[0]] == [0,24]
assert replacements[1][0][1] == expected_ring.numpy().tobytes()
worker.before_requests(output(releases=[('A',1)]))
assert worker.by_request['B'].generation == 2 and len(worker.book.snapshots) == 1
# Released pool storage can be reused; a duplicate release cannot return it twice.
old_ptr = payload.data.data_ptr()
worker.before_requests(output(releases=[('A',1)]))
assert len(worker.free_payload_indices) == 1
runner.req_states.req_id_to_index['C'] = 1
worker.after_requests(output(new=['C'], versions={'C':3}))
from vllm.v1.kv_offload.gb10_terminal_versions import Version
new_payload = worker.book.lookup(Version('C',3))
assert new_payload.data.data_ptr() == old_ptr
assert new_payload.tag.tolist() == [-1,-1,-1]
assert not worker.free_payload_indices
assert worker.book.lookup(Version('B',2)).data.data_ptr() != old_ptr
print(json.dumps(dict(passed=True, bytes_per_request=worker.byte_count, checks=[
    'actual worker descriptor binding', 'selective terminal normalization',
    'deferred snapshot survives next-forward overwrite and slot reuse',
    'resident copy uses frozen recurrent and ring data', 'disk fallback uses same version',
    'old generation release does not affect new occupant',
    'bounded payload pool reused only after release; duplicate release ignored',
])))
