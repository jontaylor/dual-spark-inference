from pathlib import Path
import json,hashlib,difflib
p=Path(__file__).resolve().parent;root=p.parents[1];old=root/'experiments/spec-determinism-20260914';base=root/'experiments/targeted-determinism-20260913/completion.fixed.py'
s=base.read_text();s=s.replace('    completion_saves: dict[str, CompletionSave] = field(default_factory=dict)','    completion_saves: dict[str, CompletionSave] = field(default_factory=dict)\n    checkpoint_requests: list[tuple[str, int]] = field(default_factory=list)\n    checkpoint_finished: set[str] = field(default_factory=set)')
s=s.replace('        boundary = min(request.num_computed_tokens, request.num_tokens - 1)','        boundary = min(request.num_computed_tokens, request.num_tokens - 1)\n        if not active:\n            # Save the latest arithmetic-grid state, replaying at most 63 tokens.\n            boundary = boundary // 64 * 64')
s=s.replace('def capture_completion_states(connector, runner, metadata):','def _capture_current_completion_states(connector, runner, metadata):')
s += '''

def capture_completion_states(connector, runner, metadata):
    """Retain only recurrent/ring state at grid checkpoints, not attention pages.

    These private CPU snapshots are bounded by live request slots. Attention
    prefix pages are immutable and copied only when the request finishes.
    """
    import torch
    if not isinstance(metadata, CompletionMetadata):
        return
    worker = connector.connector_worker.worker
    shadows = getattr(worker, "aligned_completion_shadows", None)
    if shadows is None:
        shadows = worker.aligned_completion_shadows = {}
    groups = connector._completion_groups
    attempted = getattr(worker, "aligned_completion_attempted", None)
    if attempted is None:
        attempted = worker.aligned_completion_attempted = {}
    for rid, boundary in metadata.checkpoint_requests:
        ri = runner.req_states.req_id_to_index.get(rid)
        if ri is None or attempted.get(rid) == boundary:
            continue
        attempted[rid] = boundary
        block_ids = []
        for gi, group in enumerate(groups):
            source_group = connector._completion_order[gi]
            if isinstance(group_spec(group), (MambaSpec, CircularBufferSpec)):
                # Recurrent state and ring tables use unexpanded block IDs.
                if runner.block_tables.blocks_per_kv_block[source_group] != 1:
                    raise RuntimeError("Unsupported expanded checkpoint state table")
                count = int(runner.block_tables.num_blocks.np[source_group, ri])
                ids = runner.block_tables.block_tables[source_group].gpu[ri, :count].cpu().tolist()
            else:
                ids = []
            block_ids.append(ids)
        # Private scratch ID never enters the transfer-job namespace.
        jid = -1
        save = CompletionSave(CompletionRecord(boundary + 1, b"", boundary, []), tuple(block_ids), jid)
        scratch = CompletionMetadata(load_jobs={}, store_jobs={}, completion_saves={rid: save})
        _capture_current_completion_states(connector, runner, scratch)
        if jid in connector._invalid_completions:
            connector._invalid_completions.discard(jid)
            worker.completion_replacements.pop(jid, None)
            continue
        pieces = worker.completion_replacements.pop(jid)
        for gi, group in enumerate(groups):
            if not isinstance(group_spec(group), CircularBufferSpec):
                continue
            bid = block_ids[gi][0]
            spans = []
            for ref in worker.kv_caches.group_data_refs[gi]:
                cached = worker.kv_caches.tensors[ref.tensor_idx]
                raw = cached.tensor.view(torch.uint8).reshape(-1, cached.page_size_bytes)
                data = raw[bid, :ref.page_size_bytes].cpu().contiguous().numpy().tobytes()
                spans.append((worker.tensor_offsets[ref.tensor_idx], data))
            pieces[gi] = spans
        shadows[rid] = (boundary, pieces)

    remaining = {}
    for rid, save in metadata.completion_saves.items():
        shadow = shadows.get(rid)
        if shadow is not None and shadow[0] == save.record.boundary:
            worker.completion_replacements[save.job_id] = shadow[1]
            logger.info("Completion checkpoint using aligned shadow request=%s boundary=%d", rid, save.record.boundary)
        else:
            # Exact aligned finishes and active pressure saves retain the
            # existing accepted-state validation. Missing older state is
            # rejected, never advertised as a cache hit.
            remaining[rid] = save
    if remaining:
        _capture_current_completion_states(connector, runner, CompletionMetadata(load_jobs={}, store_jobs={}, completion_saves=remaining))
    for rid in metadata.checkpoint_finished:
        shadows.pop(rid, None)
        attempted.pop(rid, None)
'''
(p/'completion.aligned.py').write_text(s)
s=(old/'scheduler.spec.py').read_text();needle='        if start >= prefill_end:\n            return num_new_tokens';assert needle in s
# Keep the original verification shape. Capture accepted states after crossing the grid.
(p/'scheduler.aligned.py').write_text(s)
s=(root/'files/paging/aligned_connector.py').read_text();needle='            meta = self._completion.add_metadata(meta)';assert needle in s
s=s.replace(needle,needle+'''\n            meta.checkpoint_requests = [
                (rid, computed // 64 * 64)
                for rid, computed, generated in zip(
                    cached.req_ids, cached.num_computed_tokens, cached.num_output_tokens
                )
                if generated > 0 and computed >= 64
            ]
            meta.checkpoint_finished = set(scheduler_output.finished_req_ids)''')
(p/'connector.aligned.py').write_text(s)
updates={'distributed/kv_transfer/kv_connector/v1/gb10_completion.py':p/'completion.aligned.py','distributed/kv_transfer/kv_connector/v1/gb10_aligned_offloading_connector.py':p/'connector.aligned.py','v1/core/sched/scheduler.py':p/'scheduler.aligned.py'}
for f in updates.values():compile(f.read_text(),str(f),'exec')
for rank in [0,1]:
 cfg=json.loads((old/f'candidate-r{rank}.json').read_text())
 for key,f in updates.items():cfg['runtime_overrides'][key]=str(f);cfg['runtime_override_sha256'][key]=hashlib.sha256(f.read_bytes()).hexdigest()
 (p/f'candidate-aligned-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
print('Aligned-checkpoint candidate built; not deployed.')
