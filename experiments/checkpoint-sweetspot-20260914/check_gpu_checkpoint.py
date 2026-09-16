"""Isolated copy oracle and timing, without loading weights or changing serving.

Timings are GPU microbenchmarks under live background inference, NOT measured
end-to-end inference gains. No live KV pointers are accessed.
"""
import json
import math
import statistics
import time
from pathlib import Path

import torch
from gpu_checkpoint import launch
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateShapeCalculator as Shapes,
)


def run(requests, shapes, repeats=5, rollback=0):
    tables = [torch.arange(1 + 5*r, 6 + 5*r, dtype=torch.int32, device='cuda')
              for r in range(requests)]
    cpu_accepted = [r % 4 + 1 for r in range(requests)]
    cpu_bias = [n - 1 - rollback for n in cpu_accepted]
    # Include a nonzero running-state column; no implicit column-zero shortcut.
    cpu_columns = [r % 2 for r in range(requests)]
    columns = torch.tensor(cpu_columns, dtype=torch.int32, device='cuda')
    computed = torch.full_like(columns, 1024)
    accepted = torch.tensor(cpu_accepted, dtype=torch.int32, device='cuda')
    boundaries = torch.full_like(columns, 1024-rollback)
    desc, sources, destinations, expected = [], [], [], []
    state_bytes = 0
    block_count = 5 * requests + 1
    for length, axis_stride, axis_length in shapes:
        # Unique data per physical source block; no serving allocations involved.
        source = torch.randint(0, 256, (block_count, length),
                               dtype=torch.uint8, device='cuda')
        sources.append(source)
        state_bytes += length
        for r in range(requests):
            dst = torch.full((length,), 173, dtype=torch.uint8, device='cuda')
            destinations.append(dst)
            desc.append([source.data_ptr(), dst.data_ptr(), length, source.stride(0),
                         block_count, tables[r].data_ptr(), 5, r, axis_stride, axis_length])
            bias = cpu_bias[r]
            if bias < 0:
                expected.append(dst.clone())
                continue
            block = 1 + 5*r + cpu_columns[r] + (bias if axis_stride == 0 else 0)
            if axis_stride:
                view = source[block].reshape(-1, axis_length, axis_stride)
                oracle = torch.zeros_like(view)
                oracle[:, :axis_length-bias].copy_(view[:, bias:])
                expected.append(oracle.flatten())
            else:
                expected.append(source[block].clone())
    descriptors = torch.tensor(desc, dtype=torch.int64, device='cuda')
    statuses = torch.zeros(len(desc), dtype=torch.int32, device='cuda')
    expected_status = torch.tensor([int(cpu_bias[d[7]] >= 0) for d in desc],
                                   dtype=torch.int32, device='cuda')
    max_bytes = max(s[0] for s in shapes)

    def copy():
        launch(descriptors, computed, accepted, columns, boundaries, statuses, max_bytes)

    copy()
    torch.cuda.synchronize()
    assert torch.equal(statuses, expected_status), 'incorrect accepted-state validity'
    assert all(torch.equal(a, b) for a, b in zip(destinations, expected))
    samples = []
    for _ in range(repeats):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); copy(); b.record(); b.synchronize()
        samples.append(a.elapsed_time(b))

    # GPU-only capture/replay gate: no data-dependent host reads in the primitive.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        copy()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(statuses, expected_status)
    assert all(torch.equal(a, b) for a, b in zip(destinations, expected))

    # Invalid/truncated accepted state must leave destinations unchanged.
    for dst in destinations:
        dst.fill_(173)
    boundaries.fill_(1010)
    copy(); torch.cuda.synchronize()
    assert not bool(statuses.any())
    assert all(bool((d == 173).all()) for d in destinations)
    boundaries.fill_(1024)
    # Missing source blocks and out-of-range columns must also be rejected.
    for table in tables:
        table.zero_()
    copy(); torch.cuda.synchronize()
    assert not bool(statuses.any())
    columns.fill_(99)
    copy(); torch.cuda.synchronize()
    assert not bool(statuses.any())
    return {'requests_in_one_launch': requests, 'pieces': len(desc), 'rollback': rollback,
            'bytes_per_request_per_rank': state_bytes,
            'copied_bytes': state_bytes*requests,
            'gpu_ms': samples, 'median_gpu_ms': statistics.median(samples),
            'byte_exact': True, 'graph_replay_exact': True,
            'invalid_state_preserves_destination': True}


if __name__ == '__main__':
    config = json.loads(Path('/model-store/snapshots/'
        'fc694b54fb0174e0913e6adf86691ef85a4ead47/config.json').read_text())['text_config']
    gdn_conv, gdn_temporal = Shapes.gated_delta_net_state_shape(
        2, config['linear_num_key_heads'], config['linear_num_value_heads'],
        config['linear_key_head_dim'], config['linear_value_head_dim'],
        config['linear_conv_kernel_dim'], 3)
    (ple_conv,) = Shapes.short_conv_state_shape(
        tp_world_size=1, intermediate_size=config['hidden_size']*config['hc_count'],
        conv_kernel=(config['ple_conv_kernel_size']-1)*config['ngram_size']+1,
        num_spec=3)
    shapes = []
    for _ in range(config['layer_types'].count('linear_attention')):
        shapes.extend([(math.prod(gdn_conv)*2, gdn_conv[1]*2, gdn_conv[0]),
                       (math.prod(gdn_temporal)*4, 0, 1)])
    for _ in config['ple_layer_ids']:
        shapes.append((math.prod(ple_conv)*2, ple_conv[1]*2, ple_conv[0]))
    started = time.time()
    # Small alternate time-axis layout checks the dim-first convolution mapping.
    small = [run(4, [(2*6*17, 2, 6), (2*6*17, 2*17, 6), (4*128, 0, 1)],
                 repeats=1, rollback=rollback) for rollback in range(5)]
    measured = [run(n, shapes) for n in (1, 4)]
    print(json.dumps({'scope': 'isolated GPU accepted-state copy, live background load; '
                     'not a serving checkpoint/reuse or throughput validation',
                     'small_layout_oracle': small, 'model_shape_copies': measured,
                     'elapsed_seconds': time.time()-started}, indent=2), flush=True)
