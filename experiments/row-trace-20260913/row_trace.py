"""Bounded, opt-in snapshots outside CUDA graph capture. No tensor mutations."""
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import torch

CONTROL = Path('/tmp/vllm-row-trace-control.json')
OUTPUT = Path('/tmp/vllm-row-trace')
_count = 0


def snapshot(x, depth=0):
    if isinstance(x, torch.Tensor):
        if x.numel() > 200000:
            return {'shape': list(x.shape), 'dtype': str(x.dtype), 'omitted': True}
        return x.detach().cpu().clone()
    if isinstance(x, np.ndarray):
        return x.copy()
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if depth > 5:
        return str(type(x))
    if isinstance(x, dict):
        return {str(k): snapshot(v, depth+1) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [snapshot(v, depth+1) for v in x]
    if dataclasses.is_dataclass(x):
        return {f.name: snapshot(getattr(x, f.name), depth+1)
                for f in dataclasses.fields(x)}
    return str(type(x))


def begin(runner, batch, desc, tables, slots, metadata):
    global _count
    runner._row_trace_pending = None
    if not CONTROL.exists():
        return
    control = json.loads(CONTROL.read_text())
    if not control.get('enabled') or _count >= control.get('max_batches', 160):
        return
    if not any('rowtrace-' in r for r in batch.req_ids):
        return
    prompt = runner.req_states.prompt_len.np[batch.idx_mapping_np]
    generated = batch.num_computed_tokens_np - prompt
    if not np.any(generated < control.get('max_generated', 4)):
        return
    _count += 1
    state = runner.req_states
    n = batch.num_reqs
    row = {
        'pid': os.getpid(), 'step': _count, 'phase': control.get('phase'),
        'batch': snapshot(batch), 'descriptor': snapshot(desc),
        'request_id_to_index': dict(state.req_id_to_index),
        'index_to_request_id': dict(state.index_to_req_id),
        'prompt_len': prompt.copy(),
        'computed_gpu': state.num_computed_tokens.gpu[batch.idx_mapping].cpu(),
        'last_sampled': state.last_sampled_tokens[batch.idx_mapping].cpu(),
        'block_tables': snapshot(tables), 'slot_mappings': snapshot(slots),
        'num_blocks': runner.block_tables.num_blocks.np[:, batch.idx_mapping_np].copy(),
        'attention_metadata': snapshot(metadata),
        'state_token_ids': [state.all_token_ids.gpu[int(idx), :int(length)+5].cpu().clone()
                            for idx, length in zip(batch.idx_mapping_np, prompt)],
        'batch_sharded_sampling': runner.batch_sharder is not None,
    }
    mamba = runner.model_state
    for name in ('_mamba_state_idx_gpu', 'num_accepted_tokens'):
        if hasattr(mamba, name):
            row[name] = snapshot(getattr(mamba, name))
    runner._row_trace_pending = row


def scores(runner, batch, global_batch, hidden_states, logits):
    row = getattr(runner, '_row_trace_pending', None)
    if row is None:
        return
    row['sampling_batch'] = snapshot(batch)
    row['sample_hidden_states'] = hidden_states.detach().cpu().clone()
    row['raw_logits'] = logits.detach().cpu().clone()
    idx = batch.idx_mapping
    states = runner.sampler.sampling_states
    row['temperature_cpu'] = states.temperature.np[batch.idx_mapping_np].copy()
    row['temperature_gpu'] = states.temperature.gpu[idx].cpu().clone()


def finish(runner, sampler_output):
    row = getattr(runner, '_row_trace_pending', None)
    if row is None:
        return
    row['sampler_output'] = snapshot(sampler_output)
    OUTPUT.mkdir(exist_ok=True)
    torch.save(row, OUTPUT / f"{row['pid']}-{row['step']:04d}.pt")
    runner._row_trace_pending = None


def _guard(fn):
    def wrapped(runner, *args):
        if getattr(runner, '_row_trace_failed', False):
            return
        try:
            return fn(runner, *args)
        except Exception:
            import traceback
            OUTPUT.mkdir(exist_ok=True)
            (OUTPUT / f'error-{os.getpid()}.txt').write_text(traceback.format_exc())
            runner._row_trace_failed = True
            runner._row_trace_pending = None
    return wrapped


begin = _guard(begin)
scores = _guard(scores)
finish = _guard(finish)
