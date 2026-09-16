"""Bounded, opt-in snapshots outside CUDA graph capture. No tensor mutations."""
import dataclasses
import hashlib
import re
import json
import os
from pathlib import Path

import numpy as np
import torch

CONTROL = Path('/tmp/vllm-row-trace-control.json')
OUTPUT = Path('/tmp/vllm-row-trace')
_count = 0
_active_row = None


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
    global _count, _active_row
    _active_row = None
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
    _active_row = row
    row['trace_control'] = control
    row['row_hashes'] = []
    row['layers'] = []
    row['full_ops'] = []
    for key in ('ngram_context', 'ple_query_start_loc'):
        if hasattr(runner.model_state, key):
            row[key] = snapshot(getattr(runner.model_state, key))
    install_layer_hooks(runner)


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
    global _active_row
    row = getattr(runner, '_row_trace_pending', None)
    _active_row = None
    if row is None:
        return
    row['sampler_output'] = snapshot(sampler_output)
    OUTPUT.mkdir(exist_ok=True)
    torch.save(row, OUTPUT / f"{row['pid']}-{row['step']:04d}.pt")
    runner._row_trace_pending = None



def install_layer_hooks(runner):
    if hasattr(runner, '_row_trace_layer_handles'):
        return
    import re
    runner._row_trace_layer_handles = []
    pattern = re.compile(r'(?:^|\.)layers\.\d+(?:\.(?:linear_attn|self_attn|mlp|ple)(?:\.ple_embedding)?)?$')

    def capture(name, event, value, module=None):
        row = getattr(runner, '_row_trace_pending', None)
        if row is None:
            return
        try:
            indices = row['batch']['logits_indices'].long()
            total = row['batch']['num_tokens_after_padding']
            def select(v):
                if isinstance(v, torch.Tensor):
                    if v.ndim and v.shape[0] == total:
                        return v.detach()[indices.to(v.device)].cpu().clone()
                    return {'shape': list(v.shape), 'dtype': str(v.dtype)}
                if isinstance(v, (tuple, list)):
                    return [select(x) for x in v]
                if isinstance(v, dict):
                    return {k: select(x) for k,x in v.items()}
                return None
            row['layers'].append({'name':name, 'event':event, 'values':select(value)})
            if row['trace_control'].get('hash_all_rows'):
                bounds = row['batch']['query_start_loc_np']
                def hash_rows(v, path=''):
                    if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == total:
                        # Inputs may include uninitialized out buffers. Keep events
                        # labelled so analysis can exclude those from attribution.
                        cpu = v.detach().cpu().contiguous()
                        hashes = [hashlib.sha256(cpu[int(a):int(b)].view(torch.uint8).numpy().tobytes()).hexdigest()
                                  for a, b in zip(bounds[:-1], bounds[1:])]
                        row['row_hashes'].append({'name': name, 'event': event,
                            'path': path, 'shape': list(v.shape[1:]), 'hashes': hashes})
                    elif isinstance(v, (list, tuple)):
                        for i, child in enumerate(v): hash_rows(child, path+'/'+str(i))
                    elif isinstance(v, dict):
                        for key, child in v.items(): hash_rows(child, path+'/'+str(key))
                hash_rows(value)

            detail_layers = row['trace_control'].get('full_layers', [0, 2, 3])
            match = re.search(r'layers\.(\d+)(?:\.|$)', name)
            full = (match and int(match.group(1)) in detail_layers
                    and (any('serial-before-0-' in r for r in row['batch']['req_ids'])
                         or row['batch']['num_reqs'] == 4)
)
            if full:
                def cpu(v):
                    if isinstance(v, torch.Tensor):return v.detach().cpu().clone()
                    if isinstance(v,(tuple,list)):return [cpu(x) for x in v]
                    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
                    return None
                item={'name':name,'event':event,'values':cpu(value)}
                if event == 'input' and name.endswith('.linear_attn'):
                    ids=row['attention_metadata'][name]['non_spec_state_indices_tensor'][:row['batch']['num_reqs']].long()
                    item['initial_cache']=[v[ids.to(v.device)].detach().cpu().clone() for v in module.kv_cache]
                if event == 'input' and hasattr(module,'weight'):
                    directory=Path('/tmp/vllm-gdn-weights');directory.mkdir(exist_ok=True)
                    file=directory/(name+'.pt')
                    if not file.exists():
                        torch.save({'weight':module.weight.detach().cpu(),
                            'bias':cpu(getattr(module,'bias',None)),
                            'class':str(type(module)),
                            'quant_method':str(type(getattr(module,'quant_method',None)))},file)
                row['full_ops'].append(item)
        except Exception as exc:
            row.setdefault('layer_errors', []).append(repr(exc))

    for name, module in runner.model.named_modules():
        if not ('.layers.' in name or name.endswith('.embed_tokens') or name.endswith('.norm')):
            continue
        def pre(module, args, kwargs, name=name):
            capture(name, 'input', {'args': args, 'kwargs': kwargs}, module)
        def post(module, args, kwargs, output, name=name):
            capture(name, 'output', output, module)
        if name.endswith('.linear_attn') and hasattr(module, '_forward_core'):
            original_core = module._forward_core
            def traced_core(*args, _fn=original_core, _name=name, _module=module, **kwargs):
                result = _fn(*args, **kwargs)
                out = kwargs.get('core_attn_out', args[3] if len(args) > 3 else None)
                capture(_name + '.core', 'output', out, _module)
                return result
            module._forward_core = traced_core
        runner._row_trace_layer_handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
        runner._row_trace_layer_handles.append(module.register_forward_hook(post, with_kwargs=True))


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


def qsa_stage(layer_name, stage, **values):
    row = _active_row
    if row is None or '.layers.3.self_attn.' not in layer_name:
        return
    try:
        indices = row['batch']['logits_indices'].long()
        full = {k: v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v
                for k, v in values.items()}
        name = layer_name.removesuffix('.attn') + '.' + stage
        row['full_ops'].append({'name': name, 'event': 'output', 'values': full})
        sample = {k: v[indices] if isinstance(v, torch.Tensor) else v
                  for k, v in full.items()}
        row['layers'].append({'name': name, 'event': 'output', 'values': sample})
    except Exception as exc:
        row.setdefault('layer_errors', []).append(repr(exc))


def force_eager():
    if not CONTROL.exists():
        return False
    control = json.loads(CONTROL.read_text())
    return bool(control.get('enabled') and control.get('force_eager'))
