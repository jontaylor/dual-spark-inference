"""Check the Triton terminal marker against the deployed scheduler, on CPU.

Run with TRITON_INTERPRET=1 inside the serving image; no model/GPU allocation.
Arguments: candidate terminal_state.py, deployed vllm scheduler utils.py.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS

assert os.environ.get('TRITON_INTERPRET') == '1', 'CPU interpreter required'
import torch

spec = importlib.util.spec_from_file_location('terminal_state', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
mark = module.make_marker()
tree = ast.parse(Path(sys.argv[2]).read_text())
f = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'check_stop')
namespace = dict(Request=object, RequestStatus=NS(
    FINISHED_STOPPED=1, FINISHED_LENGTH_CAPPED=2, FINISHED_REPETITION=3))
exec(compile(ast.Module(body=[f], type_ignores=[]), sys.argv[2], 'exec'), namespace)

rng = random.Random(719)
cases = 0
for step in range(120):
    slots, width, stop_width = 8, 4, 4
    order = list(range(slots))
    rng.shuffle(order)
    mapping = torch.tensor(order+[-1], dtype=torch.int32)
    sampled = torch.full((slots+1, width), -1, dtype=torch.int64)
    counts = torch.zeros(slots+1, dtype=torch.int32)
    after = torch.zeros(slots, dtype=torch.int32)
    prompts = torch.zeros_like(after)
    limits = torch.zeros_like(after)
    eos = torch.full_like(after, -1)
    stops = torch.full((slots, stop_width), -1, dtype=torch.int64)
    stop_counts = torch.zeros_like(after)
    boundaries = torch.full_like(after, -1)
    steps = torch.full_like(after, -1)
    expected = []
    max_model_len = 240
    for ri in range(slots):
        row = order.index(ri)
        prompt = rng.randrange(1, 200)
        before = prompt+rng.randrange(50)
        tokens = [rng.randrange(12) for _ in range(rng.randrange(width+1))]
        eos_id = rng.choice([None, 2, 3])
        stop_ids = rng.sample(range(12), rng.randrange(stop_width+1))
        limit = rng.randrange(1, 55)
        req = NS(pooling_params=None, sampling_params=NS(eos_token_id=eos_id,
            stop_token_ids=stop_ids, min_tokens=rng.randrange(8), repetition_detection=None),
            output_token_ids=[8]*(before-prompt), num_tokens=before,
            num_output_tokens=before-prompt, max_tokens=limit)
        boundary = -1
        for tok in tokens:
            req.output_token_ids.append(tok)
            req.num_tokens += 1
            req.num_output_tokens += 1
            if namespace['check_stop'](req, max_model_len):
                boundary = req.num_tokens-1
                break
        oracle = module.terminal_boundary(tokens, total_before=before,
            prompt_len=prompt, max_tokens=limit, max_model_len=max_model_len,
            eos_token_id=eos_id, stop_token_ids=stop_ids)
        assert (oracle if oracle is not None else -1) == boundary
        expected.append(boundary)
        if tokens:
            sampled[row, :len(tokens)] = torch.tensor(tokens)
        counts[row] = len(tokens)
        after[ri], prompts[ri], limits[ri] = before+len(tokens), prompt, limit
        eos[ri] = eos_id if eos_id is not None else -1
        if stop_ids:
            stops[ri, :len(stop_ids)] = torch.tensor(stop_ids)
        stop_counts[ri] = len(stop_ids)
        cases += 1
    args = (mapping, sampled, counts, after, prompts, limits, eos, stops,
            stop_counts, boundaries, steps, width, stop_width, max_model_len)
    mark[(slots+1,)](*args, step, STOP_TILE=stop_width)
    assert boundaries.tolist() == expected, (step, boundaries.tolist(), expected)
    assert steps.tolist() == [step if b >= 0 else -1 for b in expected]
    # Even if a later queued batch contains different stop positions, the
    # original terminal version must remain immutable.
    sampled.fill_(2)
    eos.fill_(2)
    counts.fill_(width)
    after += width
    mark[(slots+1,)](*args, step+1, STOP_TILE=stop_width)
    for ri, boundary in enumerate(expected):
        if boundary >= 0:
            assert boundaries[ri].item() == boundary and steps[ri].item() == step

print(json.dumps(dict(passed=True, cases=cases, mode='Triton CPU interpreter',
    checks=['matches deployed check_stop', 'empty and multi-token outputs',
            'reordered slots', 'masked rows', 'first terminal version immutable'])))
