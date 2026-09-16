"""CPU-interpreter test of the actual conditional snapshot byte-copy kernel."""
import importlib.util
import json
import os
import sys

assert os.environ.get('TRITON_INTERPRET') == '1'
import torch

spec = importlib.util.spec_from_file_location('terminal_state', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
capture = module.make_snapshot_copier()
scale = int(os.environ.get("COPY_SCALE", "1"))
assert 1 <= scale <= 4096
sources = [torch.arange(160*scale, dtype=torch.uint8).reshape(5, 32*scale) for _ in range(4)]
targets = [torch.full((24*scale,), 85, dtype=torch.uint8) for _ in sources]
tables = [torch.tensor([1, 2, 3, 4], dtype=torch.int32) for _ in sources]
statuses = [torch.full((1,), -1, dtype=torch.int32) for _ in sources]
tags = [torch.full((3,), -1, dtype=torch.int64) for _ in range(3)]
computed = torch.tensor([100, 100, 100], dtype=torch.int32)
accepted = torch.tensor([4, 4, 4], dtype=torch.int32)
columns = torch.tensor([1, 1, 1], dtype=torch.int32)
boundaries = torch.tensor([99, 90, 99], dtype=torch.int32)
terminal_steps = torch.tensor([7, 7, 6], dtype=torch.int32)
descriptors = []
for i, (ri, axis_stride, axis_length, circular, writes_tag) in enumerate([
    (0, 4*scale, 6, 0, 1), (0, 0, 1, 0, 0),
    (1, 0, 1, 1, 1), (2, 4*scale, 6, 0, 1),
]):
    descriptors.append([
        sources[i].data_ptr(), targets[i].data_ptr(), 24*scale, 32*scale, 5,
        tables[i].data_ptr(), 4, ri, axis_stride, axis_length,
        circular, statuses[i].data_ptr(), tags[ri].data_ptr(), writes_tag, 43+ri,
    ])
descriptors = torch.tensor(descriptors, dtype=torch.int64)
args = (descriptors, computed, accepted, columns, boundaries, terminal_steps)
capture[(2, 4)](*args, 7, MAX_SPEC=3, TILE=4096, COPY_LANES=2)
expected_conv = torch.cat((sources[0][2, 8*scale:24*scale], torch.zeros(8*scale, dtype=torch.uint8)))
assert torch.equal(targets[0], expected_conv)
assert torch.equal(targets[1], sources[1][4, :24*scale])
assert torch.equal(targets[2], sources[2][1, :24*scale])
assert targets[3].eq(85).all() and statuses[3].item() == -1
assert [s.item() for s in statuses[:3]] == [1, 1, 1]
assert tags[0].tolist() == [99, 7, 43]
assert tags[1].tolist() == [90, 7, 44]
assert tags[2].tolist() == [-1, -1, -1]

# Later execution destroys the live sources, but must not alter the earlier
# terminal snapshot or its immutable boundary/generation identity.
saved = [t.clone() for t in targets]
for source in sources:
    source.zero_()
computed += 4
capture[(2, 4)](*args, 8, MAX_SPEC=3, TILE=4096, COPY_LANES=2)
assert all(torch.equal(a, b) for a, b in zip(saved, targets))

# A terminal boundary whose speculative state is unavailable is rejected,
# exactly as by the original accepted-state normalization kernel.
terminal_steps[0] = 9
capture[(2, 4)](*args, 9, MAX_SPEC=3, TILE=4096, COPY_LANES=2)
assert statuses[0].item() == 0 and statuses[1].item() == 0
assert torch.equal(targets[0], saved[0]) and torch.equal(targets[1], saved[1])
print(json.dumps(dict(passed=True, mode='Triton CPU interpreter', checks=[
    'convolution normalization and zero-fill', 'temporal speculative selection',
    'circular-buffer copy', 'padded source stride', 'continuing requests untouched',
    'later source overwrite leaves snapshot intact', 'immutable generation tag',
    'unavailable truncated boundary rejected',
])))
