# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Completion-only Qwen4 snapshots using the aligned connector's disk budget."""

import hashlib
from array import array
from collections import OrderedDict
from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    TransferJobStatus,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec, LookupResult, make_offload_key
from vllm.v1.request import RequestStatus

logger = init_logger(__name__)


def prefix_digests(tokens, salt, lengths):
    """Hash candidate prefixes in one pass, even with many saved turns."""
    lengths = sorted(set(lengths))
    if not lengths:
        return {}
    h = hashlib.sha256(b"gb10-completion-v1\0")
    salt = (salt or "").encode()
    h.update(len(salt).to_bytes(8, "little"))
    h.update(salt)
    raw = memoryview(array("I", tokens[: lengths[-1]])).cast("B")
    result = {}
    start = 0
    for length in lengths:
        end = length * 4
        h.update(raw[start:end])
        result[length] = h.digest()
        start = end
    return result


def prefix_digest(tokens, salt):
    return prefix_digests(tokens, salt, [len(tokens)])[len(tokens)]


def group_spec(group):
    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec


def eligible(request):
    return not (
        request.mm_features
        or request.prompt_embeds is not None
        or request.lora_request is not None
        or request.skip_reading_prefix_cache
    )


@dataclass
class CompletionRecord:
    witness_length: int
    digest: bytes
    boundary: int
    # Transport-group index, logical page index, storage key.
    pages: list[tuple[int, int, bytes]]


@dataclass
class CompletionSave:
    record: CompletionRecord
    block_ids: tuple[list[int], ...]
    job_id: int


@dataclass
class CompletionMetadata(OffloadingConnectorMetadata):
    completion_saves: dict[str, CompletionSave] = field(default_factory=dict)
    checkpoint_requests: list[tuple[str, int]] = field(default_factory=list)
    checkpoint_finished: set[str] = field(default_factory=set)
    native_eviction_jobs: set[int] = field(default_factory=set)


@dataclass
class CompletionWorkerMetadata(OffloadingWorkerMetadata):
    invalid_completions: set[int] = field(default_factory=set)

    def aggregate(self, other):
        base = super().aggregate(other)
        assert isinstance(base, OffloadingWorkerMetadata)
        return CompletionWorkerMetadata(
            completed_jobs=base.completed_jobs,
            transfer_stats=base.transfer_stats,
            invalid_completions=self.invalid_completions
            | getattr(other, "invalid_completions", set()),
        )


class CompletionCache:
    """Scheduler-owned index; disk keys remain evictable after publication."""

    def __init__(self, connector, groups, capacity=256):
        self.connector = connector
        self.scheduler = connector.connector_scheduler
        self.groups = groups
        self.capacity = capacity
        self.records = OrderedDict()
        self.pending = {}
        self.selected = {}
        self.invalid = set()
        self.to_send = {}
        self.active = set()
        self.ready = {}
        self.cancelled = set()
        # Only pages actually loaded into this request, never token-only matches.
        self.restored_pages = {}
        self.lookup_pins = {}

    def release_lookup_pin(self, rid):
        pin = self.lookup_pins.pop(rid, None)
        if pin is not None:
            self.scheduler.manager.complete_load(*pin)

    def finish(self, request, block_ids, *, active=False, memory=False):
        s = self.scheduler
        rid = request.request_id
        self.release_lookup_pin(rid)
        self.selected.pop(rid, None)
        inherited = self.restored_pages.get(rid, {}) if active else self.restored_pages.pop(rid, {})
        state = s._req_status.get(rid)
        boundary = min(request.num_computed_tokens, request.num_tokens - 1)
        # Capture the exact accepted completion/suspension boundary on demand.
        # The restore scheduler handles its initial partial arithmetic chunk.
        if (
            state is None
            or not eligible(request)
            or not active
            and request.status
            not in (
                RequestStatus.FINISHED_STOPPED,
                RequestStatus.FINISHED_LENGTH_CAPPED,
            )
            or request.num_in_flight_tokens
            or boundary <= 0
            or not (active or memory)
            and boundary % self.connector._alignment == 0
        ):
            return False
        if rid in self.pending or rid in self.ready:
            raise RuntimeError("Snapshot already exists for request")
        state.update_offload_keys()
        witness_length = boundary + 1
        digest = prefix_digest(
            request.all_token_ids[:witness_length], request.cache_salt
        )
        pages = []
        sources = {}
        shared = []
        def shareable(candidate):
            # Suspension must release GPU capacity. Reusing a GPU-only page
            # would pin it for the parked request and defeat that purpose.
            if active and getattr(s.manager, 'native_mode', False) and s.manager.is_gpu_resident(candidate):
                return False
            return s.manager.lookup(candidate, state.req_context) == LookupResult.HIT
        # A unique namespace avoids confusing an incomplete/stale save with a hit.
        nonce = s._generate_job_id()
        for gi, group in enumerate(self.groups):
            spec = group_spec(group)
            end = (boundary + spec.block_size - 1) // spec.block_size
            if isinstance(spec, CircularBufferSpec):
                start, end = 0, 1
            elif isinstance(spec, MambaSpec):
                start = end - 1
            else:
                window = s.config.kv_group_configs[gi].sliding_window_size_in_chunks
                start = max(0, end - window - 1) if window else 0
            for index in range(start, end):
                key = None
                # An immutable full attention page loaded from a prior snapshot
                # is already the exact state needed here. Share its existing
                # storage key while resident/ready, rather than copy it anew.
                # Partial tails, recurrent state, rings and locally shared GPU
                # pages are deliberately absent from this provenance map.
                candidate = inherited.get((gi, index))
                if candidate is not None and not isinstance(spec, (MambaSpec, CircularBufferSpec)):
                    if shareable(candidate):
                        key = candidate
                        shared.append(key)
                if (key is None and memory and getattr(s.manager, 'native_mode', False)
                        and not isinstance(spec, MambaSpec)
                        and index < len(block_ids[gi])):
                    candidate = s.manager.native_key(
                        block_ids[gi][index], gi, index, state.req_context)
                    if candidate is not None:
                        key = candidate
                        shared.append(key)
                if key is None and gi < len(state.group_states) and not isinstance(spec, MambaSpec):
                    data = state.group_states[gi]
                    cfg = s.config.kv_group_configs[gi]
                    stable_end = boundary // spec.block_size - int(cfg.is_eagle_group)
                    if index < min(stable_end, len(data.offload_keys)):
                        candidate = data.offload_keys[index]
                        if shareable(candidate):
                            key = candidate
                            shared.append(key)
                if key is None:
                    # Recurrent source selection is completed on the worker before
                    # its request slot is released. It can be beyond the boundary.
                    source_index = index
                    if isinstance(spec, MambaSpec):
                        source_index = next(
                            (
                                i
                                for i in range(len(block_ids[gi]) - 1, -1, -1)
                                if block_ids[gi][i]
                            ),
                            -1,
                        )
                    if source_index < 0 or source_index >= len(block_ids[gi]):
                        return False
                    bid = block_ids[gi][source_index]
                    if bid == 0:
                        # An out-of-window page is unnecessary if already freed.
                        if (
                            not isinstance(spec, (MambaSpec, CircularBufferSpec))
                            and window is not None
                            and index < end - window
                        ):
                            continue
                        return False
                    key = make_offload_key(
                        hashlib.sha256(
                            digest
                            + nonce.to_bytes(8, "little")
                            + index.to_bytes(8, "little")
                        ).digest(),
                        gi,
                    )
                    sources[key] = bid
                pages.append((gi, index, key))
        if not sources:
            return False
        # Pin reused pages before allocating tails; the allocator must not evict
        # dependencies while this completion is being assembled.
        if shared:
            s.manager.prepare_load(shared, state.req_context)
        result = None
        if memory:
            locations = {key: (gi, index) for gi, index, key in pages if key in sources}
            borrowed = {
                key: sources[key] for gi, index, key in pages
                if key in sources and not isinstance(group_spec(self.groups[gi]), MambaSpec)
            } if getattr(s.manager, 'native_mode', False) else None
            result = s.manager.prepare_memory_store(
                list(sources), state.req_context, locations, borrowed=borrowed
            )
        if result is None:
            # A completion can go directly to disk only when resident snapshot
            # allocation cannot fit alongside the current request reservations.
            result = s.manager.prepare_store(list(sources), state.req_context)
        if result is None:
            if shared:
                s.manager.complete_load(shared, state.req_context)
            return False
        assert result.keys_to_store == list(sources)
        logger.info("Completion snapshot plan request=%s boundary=%d pages=%d reused=%d new=%d inherited=%d active=%s",
                    rid, boundary, len(pages), len(shared), len(sources),
                    sum(key in set(inherited.values()) for key in shared), active)
        record = CompletionRecord(witness_length, digest, boundary, pages)
        sizes = [0] * len(self.groups)
        indices = [0] * len(self.groups)
        for gi, index, key in pages:
            if key in sources:
                if sizes[gi] == 0:
                    indices[gi] = index
                sizes[gi] += 1
        gpu = GPULoadStoreSpec(list(sources.values()), sizes, indices)
        jid = s._generate_job_id()
        save = CompletionSave(record, block_ids, jid)
        job = TransferJob(rid, gpu, result.store_spec)
        s._jobs[jid] = TransferJobStatus(rid, s.config.num_workers, set(sources), True)
        state.transfer_jobs.add(jid)
        self.pending[rid] = (save, shared, state.req_context)
        self.to_send[rid] = (save, job)
        if active:
            self.active.add(rid)
        return True

    def add_metadata(self, meta):
        # Lookups which did not obtain an allocation this pass must not hold
        # GPU capacity while waiting for admission or a later scheduling pass.
        for rid in list(self.lookup_pins):
            self.release_lookup_pin(rid)
        saves = {}
        for rid, (save, job) in self.to_send.items():
            meta.store_jobs[save.job_id] = job
            saves[rid] = save
        self.to_send.clear()
        return CompletionMetadata(
            meta.load_jobs, meta.store_jobs, meta.jobs_to_flush, saves
        )

    def update(self, output):
        meta = output.kv_connector_worker_meta
        self.invalid.update(getattr(meta, "invalid_completions", set()))
        s = self.scheduler
        finished = set()
        for rid, (save, shared, context) in list(self.pending.items()):
            # Also drain ordinary aligned stores before releasing their GPU source.
            state = s._req_status.get(rid)
            if state is not None and state.transfer_jobs:
                continue
            record = save.record
            if rid in self.cancelled:
                self.cancelled.discard(rid)
            elif save.job_id not in self.invalid and all(
                s.manager.lookup(key, context) == LookupResult.HIT
                for _, _, key in record.pages
            ):
                if rid in self.active:
                    self.ready[rid] = record
                else:
                    key = (record.witness_length, record.digest)
                    retired = []
                    if key in self.records:
                        retired.append(self.records[key])
                    self.records[key] = record
                    self.records.move_to_end(key)
                    while len(self.records) > self.capacity:
                        retired.append(self.records.popitem(last=False)[1])
                    if retired and hasattr(s.manager, "discard_idle_keys"):
                        protected = {
                            k
                            for r in list(self.records.values())
                            + list(self.selected.values())
                            + [save.record for save, _, _ in self.pending.values()]
                            for _, _, k in r.pages
                        }
                        s.manager.discard_idle_keys(
                            {k for r in retired for _, _, k in r.pages} - protected
                        )
                logger.info(
                    "Completion checkpoint saved request=%s boundary=%d pages=%d",
                    rid,
                    record.boundary,
                    len(record.pages),
                )
            else:
                if rid in self.active:
                    raise RuntimeError("Pressure snapshot state unavailable")
                if getattr(s.manager, 'native_mode', False):
                    s.manager.discard_idle_keys({k for _, _, k in record.pages} - set(shared))
                logger.info(
                    "Completion checkpoint skipped request=%s (state unavailable)", rid
                )
            if shared:
                s.manager.complete_load(shared, context)
            self.invalid.discard(save.job_id)
            del self.pending[rid]
            if rid not in self.active:
                finished.add(rid)
        if finished:
            output.finished_sending = (output.finished_sending or set()) | finished

    def cancel_active(self, request_id):
        """Return whether cancellation must retain GPU sources until I/O drains."""
        self.active.discard(request_id)
        self.ready.pop(request_id, None)
        if request_id in self.pending:
            self.cancelled.add(request_id)
        return request_id in self.pending

    def lookup(self, request, local):
        rid = request.request_id
        self.release_lookup_pin(rid)
        self.selected.pop(rid, None)
        if not eligible(request):
            return None
        s = self.scheduler
        state = s._req_status[rid]
        if state.transfer_jobs:
            return None
        pending_records = {
            (save.record.witness_length, save.record.digest): save.record
            for rid, (save, _, _) in self.pending.items()
            if rid not in self.active and rid not in self.cancelled
        }
        lengths = {
            r.witness_length
            for r in list(self.records.values()) + list(pending_records.values())
            if local < r.boundary < request.num_prompt_tokens
        }
        digests = prefix_digests(request.all_token_ids, request.cache_salt, lengths)
        for length in sorted(lengths, reverse=True):
            key = (length, digests[length])
            record = self.records.get(key)
            if record is None:
                if key in pending_records:
                    return None, False  # matching completion is still being captured
                continue
            hits = [s.manager.lookup(k, state.req_context) for _, _, k in record.pages]
            if LookupResult.MISS in hits:
                missing = [gi for (gi, _, _), hit in zip(record.pages, hits)
                           if hit == LookupResult.MISS]
                logger.info("Completion checkpoint unavailable request=%s boundary=%d missing_groups=%s pages=%d records=%d",
                            rid, record.boundary, missing, len(record.pages), len(self.records))
                del self.records[key]
                continue
            if LookupResult.HIT_PENDING in hits:
                return None, False
            if getattr(s.manager, 'native_mode', False):
                # Full local prefix pages are adopted by the native allocator
                # before it allocates new blocks. Pin the remaining sources
                # across that allocation, then hand their lifetime to the load.
                keys = [key for gi, index, key in record.pages
                        if isinstance(group_spec(self.groups[gi]), (MambaSpec, CircularBufferSpec))
                        or index >= local // group_spec(self.groups[gi]).block_size]
                s.manager.prepare_load(keys, state.req_context)
                self.lookup_pins[rid] = (keys, state.req_context)
            self.selected[rid] = record
            self.records.move_to_end(key)
            state.num_locally_computed_tokens = local
            state.partial_tail_boundary = None
            return record.boundary - local, True
        return None

    def allocate(self, request, blocks, external):
        record = self.selected.pop(request.request_id, None)
        if record is None or external == 0:
            self.release_lookup_pin(request.request_id)
            return False
        s = self.scheduler
        state = s._req_status[request.request_id]
        assert state.num_locally_computed_tokens + external == record.boundary
        keys, ids = [], []
        restored = {}
        sizes = [0] * len(self.groups)
        indices = [0] * len(self.groups)
        for gi, index, key in record.pages:
            block = blocks.blocks[gi][index]
            if block.is_null:
                continue
            # Never overwrite a shared local prefix page. Partial tails, rings,
            # and recurrent state are allocated privately by the cache manager.
            if block.block_hash is not None:
                continue
            keys.append(key)
            ids.append(block.block_id)
            spec = group_spec(self.groups[gi])
            if not isinstance(spec, (MambaSpec, CircularBufferSpec)):
                cfg = s.config.kv_group_configs[gi]
                stable_end = record.boundary // spec.block_size - int(cfg.is_eagle_group)
                if index < stable_end:
                    restored[(gi, index)] = key
            if sizes[gi] == 0:
                indices[gi] = index
            sizes[gi] += 1
        self.restored_pages[request.request_id] = restored
        src = s.manager.prepare_load(keys, state.req_context)
        self.release_lookup_pin(request.request_id)
        jid = s._generate_job_id()
        s._current_batch_load_jobs[jid] = TransferJob(
            request.request_id, src, GPULoadStoreSpec(ids, sizes, indices)
        )
        assert not state.transfer_jobs
        state.transfer_jobs.add(jid)
        s._jobs[jid] = TransferJobStatus(
            request.request_id, s.config.num_workers, set(keys), False
        )
        for group in state.group_states:
            group.block_ids.clear()
        logger.info(
            "Completion checkpoint restore request=%s boundary=%d replay=%d",
            request.request_id,
            record.boundary,
            request.num_prompt_tokens - record.boundary,
        )
        return True


def _capture_current_completion_states(connector, runner, metadata):
    """Copy accepted recurrent state before request-slot reuse; never mutate KV.

    Attention and QSA ring pages go through the ordinary raw-page DMA. GDN
    temporal state selects a speculative block; convolution state selects a
    temporal slice. Normalize those pieces in the disk staging page so a new
    worker request can start with the ordinary neutral acceptance count of one.
    """
    import torch

    from vllm.model_executor.layers.mamba.mamba_utils import (
        get_conv_copy_spec,
        get_temporal_copy_spec,
        is_conv_state_dim_first,
    )

    if not isinstance(metadata, CompletionMetadata) or not metadata.completion_saves:
        return
    worker = connector.connector_worker.worker
    state = runner.model_state
    context = runner.vllm_config.compilation_config.static_forward_context
    groups = connector._completion_groups
    funcs = runner.model.get_mamba_state_copy_funcs(
        {
            group_spec(g).mamba_type
            for g in groups
            if isinstance(group_spec(g), MambaSpec)
        }
    )
    for rid, save in metadata.completion_saves.items():
        ri = runner.req_states.req_id_to_index.get(rid)
        if ri is None:
            connector._invalid_completions.add(save.job_id)
            continue
        # Reading the GPU scalars synchronizes the previous forward/sampling
        # stream. CPU scheduler counts alone cannot select speculative state.
        computed, accepted, column = (
            torch.stack(
                (
                    runner.req_states.num_computed_tokens.gpu[ri],
                    state.num_accepted_tokens_gpu[ri],
                    state._mamba_state_idx_gpu[ri],
                )
            )
            .cpu()
            .tolist()
        )
        bias = accepted - 1 - (computed - save.record.boundary)
        if (
            computed < save.record.boundary
            or not 0 <= bias <= runner.num_speculative_steps
        ):
            # A stop may truncate a speculative step whose in-place aligned
            # normalization has already discarded the earlier state. Keep the
            # aligned checkpoint instead of advertising a false exact hit.
            connector._invalid_completions.add(save.job_id)
            continue
        replacements = {}
        valid = True
        for gi, group in enumerate(groups):
            spec = group_spec(group)
            if not isinstance(spec, MambaSpec):
                continue
            ids = save.block_ids[gi]
            if column < 0 or column >= len(ids) or ids[column] == 0:
                valid = False
                break
            pieces = []
            for layer in group.layer_names:
                per_layer = getattr(group.kv_cache_spec, "kv_cache_specs", {}).get(
                    layer, spec
                )
                caches = context[layer].kv_cache
                copy_funcs = funcs[per_layer.mamba_type]
                if len(caches) != len(copy_funcs):
                    raise RuntimeError("Unrepresented recurrent checkpoint state")
                for tensor, copy_func in zip(caches, copy_funcs):
                    if copy_func is get_conv_copy_spec:
                        data = tensor[ids[column]].cpu().contiguous()
                        normalized = torch.zeros_like(data)
                        if is_conv_state_dim_first():
                            normalized[:, : data.shape[1] - bias].copy_(data[:, bias:])
                        else:
                            normalized[: data.shape[0] - bias].copy_(data[bias:])
                    elif copy_func is get_temporal_copy_spec:
                        if column + bias >= len(ids) or ids[column + bias] == 0:
                            valid = False
                            break
                        normalized = tensor[ids[column + bias]].cpu().contiguous()
                    else:
                        raise RuntimeError("Unsupported recurrent checkpoint copy")
                    if not tensor[0].is_contiguous():
                        raise RuntimeError("Noncontiguous recurrent checkpoint layout")
                    offset = worker.state_offset(tensor, gi)
                    pieces.append(
                        (offset, normalized.view(torch.uint8).numpy().tobytes())
                    )
                if not valid:
                    break
            if not valid:
                break
            replacements[gi] = pieces
        if valid:
            worker.completion_replacements[save.job_id] = replacements
        else:
            connector._invalid_completions.add(save.job_id)


def capture_completion_states(connector, runner, metadata):
    """Capture only completion/parking events before source slots are reused."""
    from vllm.v1.kv_offload.gb10_gpu_completion_copy import capture_gpu_completions
    remaining = capture_gpu_completions(connector, runner, metadata)
    _capture_current_completion_states(connector, runner, remaining)
