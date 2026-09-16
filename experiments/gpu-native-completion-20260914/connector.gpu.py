# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental aligned Qwen4 prefix transport, not an active-request pager.

Omit disposable compressor rings from the existing storage connector. All
scheduler metadata is projected into the remaining group namespace. Restores
must land on complete attention/Mamba boundaries; partial tails are disabled.
No production registry entry is installed: opt-in requires module_path.
"""

import hashlib
import time
from array import array
from dataclasses import dataclass, replace

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)


@dataclass(frozen=True)
class PinnedCheckpoint:
    request_id: str
    generation: int
    boundary: int
    committed_tokens: int
    token_digest: bytes
    record: object | None = None


@dataclass
class SpillJob:
    slot: int
    pending: int
    started: float


def _token_digest(request):
    return hashlib.sha256(array("I", request.all_token_ids).tobytes()).digest()


class GB10AlignedOffloadingConnector(OffloadingConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        self._pinned_checkpoints = {}
        self._completion = None
        self._invalid_completions = set()
        # The inherited filesystem tier reads only the scheduler host's mmap.
        # Remote workers have separate /dev/shm regions, even when paths match.
        transfer = getattr(vllm_config, "kv_transfer_config", None)
        extra = getattr(transfer, "kv_connector_extra_config", {})
        policy = extra.get("disk_write_policy", "eager")
        if policy not in ("eager", "pressure"):
            raise ValueError("disk_write_policy must be eager or pressure")
        self.pressure_only = policy == "pressure"
        self._pressure_saves = {}
        self.memory_completions = bool(extra.get("memory_completion_cache", False))
        self.native_completions = bool(extra.get('native_completion_cache', False))
        if self.native_completions and not self.memory_completions:
            raise ValueError('Native completion reuse requires memory completion events')
        if self.memory_completions and not self.pressure_only:
            raise ValueError("Memory completions require pressure disk policy")
        self._native_pressure = None
        self._spill_jobs = {}
        self._spill_to_send = {}
        rank_local = (
            extra.get("spec_name") == "RankLocalDiskOffloadingSpec"
            and extra.get("spec_module_path") == "vllm.v1.kv_offload.rank_local_disk"
        )
        if vllm_config.parallel_config.nnodes > 1 and not rank_local:
            raise ValueError(
                "GB10 aligned offloading requires a single host: the inherited "
                "filesystem tier cannot persist or restore remote-rank state. "
                "Multi-node use requires rank-local storage and coordinated commits."
            )
        groups = kv_cache_config.kv_cache_groups
        text = vllm_config.model_config.hf_text_config
        if getattr(text, "model_type", None) != "qwen4_exp_text":
            raise ValueError("Aligned ring omission is only validated for Qwen4")
        ratio = getattr(text, "indexer_compress_ratio", None)
        if ratio != 4:
            raise ValueError("Only compression ratio four is supported")

        def layer_specs(group):
            spec = group.kv_cache_spec
            return (
                list(spec.kv_cache_specs.values())
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else [spec]
            )

        self._keep = tuple(
            i
            for i, g in enumerate(groups)
            if not all(isinstance(s, CircularBufferSpec) for s in layer_specs(g))
        )
        if len(self._keep) == len(groups):
            raise ValueError("Expected a QSA compression ring group")
        kept = [groups[i] for i in self._keep]
        sizes = {g.kv_cache_spec.block_size for g in kept}
        if len(sizes) != 1:
            raise ValueError("Retained groups must have uniform token boundaries")
        self._alignment = sizes.pop()
        if self._alignment % ratio:
            raise ValueError("Cache boundary splits a compression group")
        for g in kept:
            for spec in layer_specs(g):
                if isinstance(spec, CircularBufferSpec):
                    raise ValueError("Mixed ring/non-ring group cannot be projected")
                if isinstance(spec, MambaSpec) and spec.mamba_cache_mode != "align":
                    raise ValueError("Mamba align mode is required")
        self._index = {old: new for new, old in enumerate(self._keep)}
        self._completion_enabled = self.pressure_only or bool(
            extra.get("completion_checkpoints", False)
        )
        self._completion_groups = kept + [
            g for i, g in enumerate(groups) if i not in self._keep
        ]
        self._completion_order = self._keep + tuple(
            i for i in range(len(groups)) if i not in self._keep
        )
        if self._completion_enabled and (
            not rank_local
            or not vllm_config.use_v2_model_runner
            or vllm_config.scheduler_config.async_scheduling
            or vllm_config.parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError(
                "Completion checkpoints require synchronous V2 TP disk offload"
            )
        super().__init__(
            vllm_config, role, replace(kv_cache_config, kv_cache_groups=kept)
        )
        if self.connector_scheduler is not None:
            scheduler = self.connector_scheduler
            scheduler.automatic_stores_enabled = not self.pressure_only
            scheduler.config = scheduler.config._replace(supports_partial_tail=False)
            scheduler._partial_tail_block_size = 0
            if self._completion_enabled:
                from .gb10_completion import CompletionCache

                self._completion = CompletionCache(self, self._completion_groups)
        if self._completion_enabled and self.connector_worker is not None:
            # Active jobs retain the first five group IDs; only completion jobs
            # use the appended ring group. All groups share the same page pool.
            self.connector_worker.kv_cache_config = replace(
                kv_cache_config, kv_cache_groups=self._completion_groups
            )

    def bind_gpu_block_pool(self, pool):
        super().bind_gpu_block_pool(pool)
        if self.pressure_only and (not self.memory_completions or self.native_completions):
            from vllm.v1.kv_offload.gb10_native_pressure import NativePressureCache

            self._native_pressure = NativePressureCache(self, pool)

    def _project(self, blocks):
        return tuple(blocks[i] for i in self._keep)

    def prepare_pressure_checkpoint(self, request, blocks, generation):
        """Capture a quiesced resident request; the caller retains all sources."""
        from vllm.v1.kv_offload.rank_local_disk import DiskSlotManager

        rid = request.request_id
        if request.num_in_flight_tokens:
            return None
        cache = self._completion
        scheduler = self.connector_scheduler
        assert self.pressure_only and cache is not None and scheduler is not None
        assert isinstance(scheduler.manager, DiskSlotManager)
        if rid not in self._pressure_saves:
            # Only one pressure save may own staging/disk allocation at a time.
            if self._pressure_saves:
                return None
            reordered = tuple(blocks[i] for i in self._completion_order)
            if not cache.finish(request, reordered, active=True):
                raise RuntimeError("Cannot capture resident pressure snapshot")
            self._pressure_saves[rid] = (generation, _token_digest(request))
            return None
        saved_generation, digest = self._pressure_saves[rid]
        if saved_generation != generation or digest != _token_digest(request):
            raise RuntimeError("Request changed while pressure snapshot was saving")
        record = cache.ready.get(rid)
        if record is None:
            return None
        context = scheduler._req_status[rid].req_context
        pinned = scheduler.manager.pin_checkpoint(
            rid, generation, [key for _, _, key in record.pages], context
        )
        if pinned is None:
            raise RuntimeError("Pressure snapshot lost before pinning")
        checkpoint = PinnedCheckpoint(
            rid,
            generation,
            record.boundary,
            min(request.num_tokens, request.num_computed_tokens),
            digest,
            record,
        )
        self._pinned_checkpoints[rid] = checkpoint
        cache.ready.pop(rid)
        cache.active.discard(rid)
        del self._pressure_saves[rid]
        return checkpoint

    def pressure_save_started(self, request_id):
        return request_id in self._pressure_saves

    def configure_completion_memory(self, pool, ledger):
        from vllm.v1.kv_offload.rank_local_disk import DiskSlotManager

        scheduler = self.connector_scheduler
        assert self.memory_completions and scheduler is not None
        assert isinstance(scheduler.manager, DiskSlotManager)
        scheduler.manager.configure_memory(pool, ledger)
        scheduler.manager.native_mode = self.native_completions

    def reclaim_completion_memory(self, required_free):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
            TransferJob,
        )
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.rank_local_disk import DiskSlotManager, DiskSlots

        scheduler = self.connector_scheduler
        assert scheduler is not None and isinstance(scheduler.manager, DiskSlotManager)
        manager = scheduler.manager
        if self.native_completions:
            # Completed pages are already in the native free/LRU pool. Only
            # actual allocator reuse can initiate their backing-store writes.
            return False
        assert manager.memory_ledger is not None
        if manager.memory_ledger.free >= required_free:
            return False
        if self._spill_jobs:
            return True
        needed = min(32, required_free - manager.memory_ledger.free)
        for slot, page in manager.begin_spill(needed):
            sizes = [0] * len(self._completion_groups)
            indices = [0] * len(sizes)
            sizes[page.group], indices[page.group] = 1, page.index
            jid = scheduler._generate_job_id()
            self._spill_jobs[jid] = SpillJob(
                slot,
                scheduler.config.num_workers,
                time.monotonic(),
            )
            self._spill_to_send[jid] = TransferJob(
                "gb10-memory-spill",
                GPULoadStoreSpec([page.block.block_id], sizes, indices),
                DiskSlots([slot]),
            )
        # Loads/captures can temporarily pin all resident pages. Wait for their
        # ACKs instead of parking an active request while those copies drain.
        return bool(self._spill_jobs or manager.resident)

    def has_pending_push_work(self):
        return bool(self._spill_jobs or (self._native_pressure and self._native_pressure.jobs)) or super().has_pending_push_work()

    def pin_request_checkpoint(self, request, committed, generation, max_replay=3200):
        """Find and pin a complete bounded-tail snapshot after output drainage.

        This does not preempt or free GPU state. The scheduler must first
        stop scheduling the request and reconcile all in-flight outputs.
        A missing/recently evicted disk block leaves the request resident.
        """
        from vllm.v1.kv_offload.rank_local_disk import DiskSlotManager

        scheduler = self.connector_scheduler
        if scheduler is None or not isinstance(scheduler.manager, DiskSlotManager):
            raise ValueError("Checkpoint pinning requires rank-local disk transport")
        if committed <= 0 or max_replay < 0:
            raise ValueError("Invalid committed token count or replay bound")
        if request.request_id in self._pinned_checkpoints:
            raise ValueError("Request already has a pinned checkpoint")
        if request.num_in_flight_tokens:
            return None
        if committed > min(request.num_tokens, request.num_computed_tokens):
            raise ValueError(
                "Checkpoint cannot include uncomputed or unaccepted tokens"
            )
        state = scheduler._req_status[request.request_id]
        state.update_offload_keys()
        boundary = (committed - 1) // self._alignment * self._alignment
        if any(g.is_eagle_group for g in scheduler.config.kv_group_configs):
            boundary -= self._alignment
        while boundary > 0 and committed - boundary <= max_replay:
            keys = []
            for group, data in zip(
                scheduler.config.kv_group_configs, state.group_states
            ):
                end = boundary // group.tokens_per_chunk
                window = group.sliding_window_size_in_chunks
                start = max(0, end - window) if window is not None else 0
                if len(data.offload_keys) < end:
                    break
                # The last speculative block depends on the following token.
                # A shared prefix key alone does not prove who supplied it.
                # Only this request's own saved boundary can use the bound
                # accepted-sequence shortcut; ordinary reuse stays conservative.
                if group.is_eagle_group and not scheduler.manager.stored_by(
                    data.offload_keys[end - 1], request.request_id
                ):
                    break
                keys.extend(data.offload_keys[start:end])
            else:
                pinned = scheduler.manager.pin_checkpoint(
                    request.request_id, generation, keys, state.req_context
                )
                if pinned is not None:
                    checkpoint = PinnedCheckpoint(
                        request.request_id,
                        generation,
                        boundary,
                        committed,
                        _token_digest(request),
                    )
                    self._pinned_checkpoints[request.request_id] = checkpoint
                    return checkpoint
            boundary -= self._alignment
        return None

    def release_request_checkpoint(self, checkpoint):
        """Caller must drain restore/cancel I/O before releasing its pin."""
        from vllm.v1.kv_offload.rank_local_disk import DiskSlotManager

        if self._pinned_checkpoints.get(checkpoint.request_id) != checkpoint:
            raise ValueError("Stale checkpoint release")
        scheduler = self.connector_scheduler
        assert scheduler is not None
        assert isinstance(scheduler.manager, DiskSlotManager)
        scheduler.manager.release_checkpoint(
            checkpoint.request_id, checkpoint.generation
        )
        del self._pinned_checkpoints[checkpoint.request_id]

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        if self._completion is not None:
            reordered = KVCacheBlocks(
                tuple(blocks.blocks[i] for i in self._completion_order)
            )
            if self._completion.allocate(request, reordered, num_external_tokens):
                return
        return super().update_state_after_alloc(
            request, KVCacheBlocks(self._project(blocks.blocks)), num_external_tokens
        )

    def build_connector_meta(self, scheduler_output):
        cached = scheduler_output.scheduled_cached_reqs
        state = scheduler_output.kv_connector_block_state
        if state is not None:
            state = replace(
                state,
                block_ids={r: self._project(b) for r, b in state.block_ids.items()},
                boundary_state_offloads={
                    r: [
                        (self._index[g], block, tokens)
                        for g, block, tokens in offers
                        if g in self._index
                    ]
                    for r, offers in state.boundary_state_offloads.items()
                },
            )
        projected = replace(
            scheduler_output,
            scheduled_new_reqs=[
                replace(r, block_ids=self._project(r.block_ids))
                for r in scheduler_output.scheduled_new_reqs
            ],
            scheduled_cached_reqs=replace(
                cached,
                new_block_ids=[
                    self._project(b) if b is not None else None
                    for b in cached.new_block_ids
                ],
            ),
            kv_connector_block_state=state,
        )
        meta = super().build_connector_meta(projected)
        if self._completion is not None:
            meta = self._completion.add_metadata(meta)
        if self._spill_to_send:
            from .offloading.common import OffloadingConnectorMetadata

            assert isinstance(meta, OffloadingConnectorMetadata)
            meta.store_jobs.update(self._spill_to_send)
        self._spill_to_send.clear()
        if any(
            time.monotonic() - job.started > 300 for job in self._spill_jobs.values()
        ):
            raise RuntimeError("Resident completion spill timed out")
        if self._native_pressure is not None:
            self._native_pressure.add_metadata(meta)
        return meta

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        checkpoint = self._pinned_checkpoints.get(getattr(request, "request_id", None))
        if checkpoint is not None:
            # Generic prefix lookup needs a following hash/chunk as a witness
            # for speculative state. Same-process suspension instead binds the
            # entire accepted sequence, including the token after the boundary.
            if _token_digest(request) != checkpoint.token_digest:
                raise RuntimeError(
                    "Accepted tokens changed while checkpoint was pinned"
                )
            if request.skip_reading_prefix_cache:
                raise RuntimeError(
                    "Pinned resume conflicts with bypassing cached state"
                )
            if not 0 <= num_computed_tokens <= checkpoint.boundary:
                raise RuntimeError("Local prefix exceeds pinned resume boundary")
            scheduler = self.connector_scheduler
            assert scheduler is not None
            state = scheduler._req_status[request.request_id]
            if state.transfer_jobs:
                return None, False
            if checkpoint.record is not None:
                assert self._completion is not None
                self._completion.selected[request.request_id] = checkpoint.record
                state.num_locally_computed_tokens = num_computed_tokens
                state.partial_tail_boundary = None
                return checkpoint.boundary - num_computed_tokens, True
            for group in state.group_states:
                group.block_ids.clear()
            state.update_offload_keys()
            state.num_locally_computed_tokens = num_computed_tokens
            state.partial_tail_boundary = None
            state.update_num_hit_chunks(checkpoint.boundary)
            scheduler._touch(state)
            count = checkpoint.boundary - num_computed_tokens
            return count, bool(count)
        if self._completion is not None:
            hit = self._completion.lookup(request, num_computed_tokens)
            if hit is not None:
                return hit
        count, pending = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        if count and (num_computed_tokens + count) % self._alignment:
            raise RuntimeError(
                "Refusing restore across an incomplete compression group"
            )
        return count, pending

    def request_finished_all_groups(self, request, block_ids):
        delayed = False
        if self.pressure_only:
            assert self._completion is not None
            delayed = self._completion.cancel_active(request.request_id)
            self._pressure_saves.pop(request.request_id, None)
            if self.memory_completions and not delayed:
                delayed = self._completion.finish(
                    request,
                    tuple(block_ids[i] for i in self._completion_order),
                    memory=True,
                )
        elif self._completion is not None:
            delayed = self._completion.finish(
                request, tuple(block_ids[i] for i in self._completion_order)
            )
        if self._completion is not None:
            self._completion.restored_pages.pop(request.request_id, None)
        old_delayed, params = super().request_finished_all_groups(request, block_ids)
        return delayed or old_delayed, params

    def update_connector_output(self, output):
        from vllm.v1.kv_offload.rank_local_disk import DiskSlotManager

        meta = output.kv_connector_worker_meta
        if meta is not None and self._native_pressure is not None:
            meta = replace(meta, completed_jobs=self._native_pressure.consume_completions(meta))
            output.kv_connector_worker_meta = meta
        if meta is not None and self._spill_jobs:
            completed = dict(meta.completed_jobs)
            for jid in list(self._spill_jobs):
                count = completed.pop(jid, 0)
                self._spill_jobs[jid].pending -= count
                assert self._spill_jobs[jid].pending >= 0
                if self._spill_jobs[jid].pending == 0:
                    scheduler = self.connector_scheduler
                    assert scheduler is not None
                    assert isinstance(scheduler.manager, DiskSlotManager)
                    scheduler.manager.finish_spill(self._spill_jobs[jid].slot)
                    del self._spill_jobs[jid]
            output.kv_connector_worker_meta = replace(meta, completed_jobs=completed)
        super().update_connector_output(output)
        if self._completion is not None:
            self._completion.update(output)

    def prepare_completion(self, runner, metadata):
        if getattr(metadata, "native_eviction_jobs", None):
            # Drain before finish/free/add/update_requests, which can zero or
            # COW into the physical blocks just allocated by the scheduler.
            self.connector_worker.handle_preemptions(metadata)
            metadata.jobs_to_flush = None
            metadata.native_eviction_jobs.clear()
        if self._completion_enabled:
            from vllm.distributed.kv_transfer.kv_connector.v1.gb10_completion import (
                capture_completion_states,
            )

            capture_completion_states(self, runner, metadata)

    def build_connector_worker_meta(self):
        meta = super().build_connector_worker_meta()
        if not self._completion_enabled:
            return meta
        from vllm.distributed.kv_transfer.kv_connector.v1.gb10_completion import (
            CompletionWorkerMetadata,
        )

        if meta is None and not self._invalid_completions:
            return None
        result = CompletionWorkerMetadata(invalid_completions=self._invalid_completions)
        if meta is not None:
            result.completed_jobs = meta.completed_jobs
            result.transfer_stats = meta.transfer_stats
        self._invalid_completions = set()
        return result

    def reset_cache(self):
        if self._native_pressure is not None and self._native_pressure.jobs:
            return False
        if self._pressure_saves or self._pinned_checkpoints or self._spill_jobs:
            return False
        if self._completion is not None:
            if self._completion.pending or self._completion.selected:
                return False
            self._completion.records.clear()
            self._completion.restored_pages.clear()
        return super().reset_cache()
