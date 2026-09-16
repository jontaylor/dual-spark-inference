"""Worker binding for selectively preserved async terminal state.

Candidate only. The scheduler bounds outstanding versions, including deferred
completions; packed buffers contain meaningful tensor bytes, not KV padding.
"""
from dataclasses import dataclass

import torch
import triton

from vllm.v1.kv_offload.gb10_terminal_state import make_marker, make_snapshot_copier
from vllm.v1.kv_offload.gb10_terminal_versions import TerminalVersions, Version


@dataclass
class Piece:
    group: int
    source: torch.Tensor
    table: torch.Tensor
    offset: int
    size: int
    axis_stride: int
    axis_length: int
    circular: bool


@dataclass
class Payload:
    data: torch.Tensor
    tag: torch.Tensor
    statuses: torch.Tensor


class TerminalWorker:
    def __init__(self, connector, runner):
        from vllm.distributed.kv_transfer.kv_connector.v1.gb10_completion import group_spec
        from vllm.v1.kv_cache_interface import MambaSpec, CircularBufferSpec
        from vllm.model_executor.layers.mamba.mamba_utils import (
            get_conv_copy_spec, get_temporal_copy_spec, is_conv_state_dim_first,
        )
        self.connector = connector
        self.runner = runner
        self.book = TerminalVersions(runner.max_num_reqs)
        self.by_request = {}
        self.sampling = {}
        self.mark = make_marker()
        self.copy = make_snapshot_copier()
        self.step = 0
        self.pieces = []
        self.byte_count = 0
        groups = connector._completion_groups
        funcs = runner.model.get_mamba_state_copy_funcs({
            group_spec(g).mamba_type for g in groups if isinstance(group_spec(g), MambaSpec)
        })
        context = runner.vllm_config.compilation_config.static_forward_context
        for gi, group in enumerate(groups):
            spec = group_spec(group)
            if not isinstance(spec, (MambaSpec, CircularBufferSpec)):
                continue
            original_group = connector._completion_order[gi]
            table = runner.block_tables.block_tables[original_group].gpu
            for layer in group.layer_names:
                per_layer = getattr(group.kv_cache_spec, 'kv_cache_specs', {}).get(layer, spec)
                caches = context[layer].kv_cache
                if isinstance(caches, torch.Tensor):
                    caches = (caches,)
                circular = isinstance(per_layer, CircularBufferSpec)
                copy_funcs = (None,)*len(caches) if circular else funcs[per_layer.mamba_type]
                if len(caches) != len(copy_funcs):
                    raise RuntimeError('Unrepresented terminal snapshot state')
                for tensor, copy_func in zip(caches, copy_funcs):
                    if not tensor[0].is_contiguous() or tensor.shape[0] != runner.kv_cache_config.num_blocks:
                        raise RuntimeError('Unsupported terminal snapshot block layout')
                    if circular or copy_func is get_temporal_copy_spec:
                        axis_stride, axis_length = 0, 1
                    elif copy_func is get_conv_copy_spec:
                        axis = 1 if is_conv_state_dim_first() else 0
                        axis_stride = tensor[0].stride(axis)*tensor.element_size()
                        axis_length = tensor[0].shape[axis]
                    else:
                        raise RuntimeError('Unsupported terminal snapshot normalization')
                    size = tensor[0].numel()*tensor.element_size()
                    self.pieces.append(Piece(gi, tensor, table, self.byte_count, size,
                                             axis_stride, axis_length, circular))
                    self.byte_count += size
        if not self.pieces:
            raise RuntimeError('Terminal snapshot has no recurrent/ring pieces')
        maximum = int(runner.vllm_config.kv_transfer_config.kv_connector_extra_config[
            'async_terminal_snapshot_bytes'])
        self.maximum = maximum
        self.fixed_bytes = runner.max_num_reqs*(self.byte_count + 24 + 4*len(self.pieces) + 20)
        if self.fixed_bytes > maximum:
            raise RuntimeError('Terminal snapshot content exceeds reserved memory budget')
        device = runner.device
        capacity = runner.max_num_reqs
        # Allocate bounded storage before serving; retired request slots do not
        # return a payload slot until the generation-qualified release arrives.
        self.payload_data = torch.empty((capacity, self.byte_count), dtype=torch.uint8, device=device)
        self.payload_tags = torch.empty((capacity, 3), dtype=torch.int64, device=device)
        self.payload_statuses = torch.empty((capacity, len(self.pieces)), dtype=torch.int32, device=device)
        self.payload_indices = {}
        self.free_payload_indices = list(range(capacity-1, -1, -1))
        self.boundaries = torch.full((runner.max_num_reqs,), -1, dtype=torch.int32, device=device)
        self.steps = torch.full_like(self.boundaries, -1)
        self.limits = torch.zeros_like(self.boundaries)
        self.eos = torch.full_like(self.boundaries, -1)
        self.stop_counts = torch.zeros_like(self.boundaries)
        self.stop_tokens = torch.full((runner.max_num_reqs, 1), -1, dtype=torch.int64, device=device)
        self.descriptors = None

    def before_requests(self, scheduler_output):
        meta = scheduler_output.kv_connector_metadata
        retiring = set(scheduler_output.finished_req_ids)
        retiring.update(scheduler_output.preempted_req_ids or ())
        for rid in retiring:
            version = self.by_request.pop(rid, None)
            if version is not None:
                self.book.retire_slot(version)
                self.sampling.pop(version, None)
        for rid, generation in getattr(meta, 'terminal_releases', ()):
            version = Version(rid, generation)
            snapshot = self.book.snapshots.get(version)
            if snapshot is not None and snapshot.active:
                # Scheduler release is generation-qualified and follows the
                # connector's all-rank completion acknowledgement.
                self.book.retire_slot(version)
                if self.by_request.get(rid) == version:
                    del self.by_request[rid]
                self.sampling.pop(version, None)
            if self.book.release(version):
                self.free_payload_indices.append(self.payload_indices.pop(version))
        if retiring or getattr(meta, 'terminal_releases', ()):
            self._rebuild()

    def after_requests(self, scheduler_output):
        versions = getattr(scheduler_output.kv_connector_metadata, 'terminal_versions', {})
        changed = False
        for request in scheduler_output.scheduled_new_reqs:
            rid = request.req_id
            version = Version(rid, versions[rid])
            slot = self.runner.req_states.req_id_to_index[rid]
            if version in self.book.snapshots:
                if self.book.snapshots[version].slot != slot:
                    raise RuntimeError('Resumed request needs a new terminal generation')
                continue
            if not self.free_payload_indices:
                raise RuntimeError('Scheduler exceeded retained terminal capacity')
            index = self.free_payload_indices[-1]
            payload = Payload(self.payload_data[index], self.payload_tags[index], self.payload_statuses[index])
            self.book.register(version, slot, payload)
            self.free_payload_indices.pop()
            self.payload_indices[version] = index
            payload.tag.fill_(-1)
            payload.statuses.fill_(-1)
            self.by_request[rid] = version
            self.sampling[version] = request.sampling_params
            self.boundaries[slot] = -1
            self.steps[slot] = -1
            changed = True
        if changed:
            self._rebuild()

    def _rebuild(self):
        active = list(self.book.active())
        width = triton.next_power_of_2(max([1]+[
            len(self.sampling[s.version].stop_token_ids or ()) for s in active]))
        # Allow old/new descriptor and stop arrays to coexist during replacement.
        scratch_bytes = 2*self.runner.max_num_reqs*(width*8 + len(self.pieces)*15*8)
        if self.fixed_bytes + scratch_bytes > self.maximum:
            raise RuntimeError('Terminal descriptors exceed reserved memory budget')
        on_cuda = self.runner.device.type == 'cuda'
        # Lifecycle updates must not wait for queued GPU work. Pinned staging
        # permits stream-ordered copies; PyTorch retains each source allocation
        # until its asynchronous copy completes.
        host = dict(device='cpu', pin_memory=on_cuda)
        stops = torch.full((self.runner.max_num_reqs, width), -1, dtype=torch.int64, **host)
        eos = torch.full((self.runner.max_num_reqs,), -1, dtype=torch.int32, **host)
        limits = torch.zeros((self.runner.max_num_reqs,), dtype=torch.int32, **host)
        counts = torch.zeros((self.runner.max_num_reqs,), dtype=torch.int32, **host)
        descriptors = []
        for snapshot in active:
            slot, payload = snapshot.slot, snapshot.payload
            params = self.sampling[snapshot.version]
            ids = params.stop_token_ids or ()
            if ids:
                stops[slot, :len(ids)] = torch.tensor(ids, device="cpu")
            eos[slot] = params.eos_token_id if params.eos_token_id is not None else -1
            limits[slot] = params.max_tokens
            counts[slot] = len(ids)
            for i, piece in enumerate(self.pieces):
                descriptors.append([
                    piece.source.data_ptr(), payload.data.data_ptr()+piece.offset,
                    piece.size, piece.source.stride(0)*piece.source.element_size(),
                    piece.source.shape[0], piece.table[slot].data_ptr(), piece.table.shape[1],
                    slot, piece.axis_stride, piece.axis_length, int(piece.circular),
                    payload.statuses[i:i+1].data_ptr(), payload.tag.data_ptr(), int(i == 0),
                    snapshot.version.generation,
                ])
        self.eos.copy_(eos, non_blocking=on_cuda)
        self.limits.copy_(limits, non_blocking=on_cuda)
        self.stop_counts.copy_(counts, non_blocking=on_cuda)
        self.stop_tokens = stops.to(self.runner.device, non_blocking=on_cuda)
        self.descriptors = (torch.tensor(descriptors, dtype=torch.int64, **host)
                            .to(self.runner.device, non_blocking=on_cuda)
                            if descriptors else None)

    def capture(self, input_batch, sampled, counts):
        if self.descriptors is None or input_batch.num_reqs == 0:
            return
        self.step += 1
        r = self.runner
        self.mark[(input_batch.num_reqs,)](
            input_batch.idx_mapping, sampled, counts, r.req_states.total_len.gpu,
            r.req_states.prompt_len.gpu, self.limits, self.eos, self.stop_tokens,
            self.stop_counts, self.boundaries, self.steps, sampled.stride(0),
            self.stop_tokens.shape[1], r.max_model_len, self.step,
            STOP_TILE=self.stop_tokens.shape[1],
        )
        self.copy[(4, self.descriptors.shape[0])](
            self.descriptors, r.req_states.num_computed_tokens.gpu,
            r.model_state.num_accepted_tokens_gpu, r.model_state._mamba_state_idx_gpu,
            self.boundaries, self.steps, self.step, MAX_SPEC=r.num_speculative_steps,
            TILE=4096, COPY_LANES=4,
        )

    def selected(self, save):
        payload = self.book.lookup(Version(save.request_id, save.terminal_version))
        # This event-only read also fences all earlier stream copies. It is
        # outside decode's common path, like the existing completion validity read.
        tag = payload.tag.cpu().tolist()
        if tag[0] != save.record.boundary or tag[1] < 0 or tag[2] != save.terminal_version:
            return None
        if not bool(payload.statuses.eq(1).all().item()):
            return None
        return payload

    def resident(self, save, job):
        payload = self.selected(save)
        if payload is None:
            return False
        destinations = {}
        cursor = 0
        for gi, count in enumerate(job.src_spec.group_sizes):
            if any(p.group == gi for p in self.pieces):
                if count != 1:
                    raise RuntimeError('Terminal snapshot requires one destination per state group')
                destinations[gi] = job.dst_spec.memory_blocks[cursor]
            cursor += count
        for piece in self.pieces:
            target = piece.source[destinations[piece.group]].view(torch.uint8).reshape(-1)
            target.copy_(payload.data[piece.offset:piece.offset+piece.size])
        return True

    def disk_replacements(self, save):
        payload = self.selected(save)
        if payload is None:
            return None
        worker = self.connector.connector_worker.worker
        replacements = {}
        raw = payload.data.cpu().numpy().tobytes()
        for piece in self.pieces:
            offset = worker.state_offset(piece.source, piece.group)
            replacements.setdefault(piece.group, []).append(
                (offset, raw[piece.offset:piece.offset+piece.size]))
        return replacements
