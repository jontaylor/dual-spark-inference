# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental same-process disk transport with rank-local staging.

The scheduler owns slot allocation; workers own files. A slot becomes readable
only through the existing all-worker store completion protocol. This is a
transport prototype, not restart recovery or an active-request parking policy.
"""

import hashlib
import json
import os
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    GPULoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingGaugeMetadata,
    OffloadingSpec,
    OffloadingWorker,
    ReqContext,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.tiering.fs.io import batch_load_block, batch_store_block

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.sched.parking_policy import ParkingPolicy


class DiskSlots(BlockIDsLoadStoreSpec):
    """Scheduler-assigned disk slots, identical IDs across worker ranks."""


class ResidentSlots(DiskSlots):
    """A slot is backed by a GPU pool page, or by disk when its page is -1."""

    def __init__(self, block_ids, memory_blocks, direct_gpu=False):
        super().__init__(block_ids)
        self.memory_blocks = memory_blocks
        self.direct_gpu = direct_gpu


@dataclass
class ResidentPage:
    block: KVCacheBlock
    key: bytes
    group: int
    index: int
    context: ReqContext
    native: bool = False
    owned: bool = True
    reserved: bool = False
    load_pins: int = 0


class DiskSlotManager(CPUOffloadingManager):
    """Reuse allocation/refcounts, without allocating a CPU cache pool."""

    def __init__(self, num_blocks):
        super().__init__(num_blocks=num_blocks)
        self.medium = Medium.STORAGE
        self._checkpoint_pins = {}
        self._store_owners = {}
        self.resident = OrderedDict()
        self.spilling = set()
        self.memory_pool: BlockPool | None = None
        self.memory_ledger: ParkingPolicy | None = None
        self.native_mode = False
        self._resident_by_block = {}
        self._detached_sources = {}

    def configure_memory(self, pool, ledger):
        self.memory_pool, self.memory_ledger = pool, ledger
        pool.native_completion_block_ids = set()

    def _free_block(self, block):
        self._release_resident(block.block_id)
        super()._free_block(block)

    def _release_resident(self, slot):
        page = self.resident.pop(slot, None)
        if page is not None:
            assert slot not in self.spilling
            assert self.memory_pool is not None and self.memory_ledger is not None
            assert page.load_pins == 0
            if not page.native or page.owned:
                self.memory_pool.free_blocks([page.block])
            if not page.native or page.reserved:
                self.memory_ledger.cache_reserved -= 1
            slots = self._resident_by_block.get(page.block.block_id)
            if slots is not None:
                slots.discard(slot)
                if not slots:
                    del self._resident_by_block[page.block.block_id]
                    self.memory_pool.native_completion_block_ids.discard(page.block.block_id)

    def prepare_memory_store(self, keys, context, locations, borrowed=None):
        """Allocate within the shared physical/ledger budget, before any copy."""
        if self.native_mode:
            return self._prepare_native_store(keys, context, locations, borrowed or {})
        needed = sum(self._policy.get(key) is None for key in keys)
        if self.memory_pool is None or self.memory_ledger is None:
            raise RuntimeError("Resident completion budget was not configured")
        if (
            self.memory_pool is None
            or needed > self.memory_ledger.free
            or needed > self.memory_pool.get_num_free_blocks()
        ):
            return None
        result = self.prepare_store(keys, context)
        if result is None:
            return None
        blocks = self.memory_pool.get_new_blocks(len(result.keys_to_store))
        self.memory_ledger.cache_reserved += len(blocks)
        assert isinstance(result.store_spec, DiskSlots)
        slots = result.store_spec.block_ids.tolist()
        for key, slot, block in zip(result.keys_to_store, slots, blocks):
            group, index = locations[key]
            self.resident[slot] = ResidentPage(block, key, group, index, context)
        result.store_spec = ResidentSlots(slots, [b.block_id for b in blocks])
        return result

    def native_key(self, block_id, group, index, context):
        """Share an existing immutable page while its allocator identity lives."""
        for slot in self._resident_by_block.get(block_id, ()):
            page = self.resident[slot]
            if (page.group == group and page.index == index
                    and self.lookup(page.key, context) == LookupResult.HIT):
                return page.key
        return None

    def is_gpu_resident(self, key):
        status = self._policy.get(key)
        return status is not None and status.block_id in self.resident

    def _prepare_native_store(self, keys, context, locations, borrowed):
        pool, ledger = self.memory_pool, self.memory_ledger
        if pool is None or ledger is None:
            raise RuntimeError('Native completion pool is not configured')
        needed = sum(k not in borrowed for k in keys)
        if needed > ledger.free:
            return None
        gate = getattr(pool, 'native_allocation_gate', None)
        if gate is not None and needed <= pool.get_num_free_blocks() + len(gate.__self__.source_blocks):
            gate(needed, request_id=context.req_id)
        if needed > pool.get_num_free_blocks():
            return None
        # Borrowed pages are still owned by the completing request. Pin them
        # before slot eviction/allocation can trigger other ownership changes.
        borrowed_blocks = {k: pool.blocks[bid] for k, bid in borrowed.items()}
        for block in borrowed_blocks.values():
            assert not block.is_null and block.ref_cnt > 0
        pool.touch(list(borrowed_blocks.values()))
        result = self.prepare_store(keys, context)
        if result is None:
            pool.free_blocks(list(borrowed_blocks.values()))
            return None
        assert result.keys_to_store == list(keys), 'New completion keys must be unique'
        allocated = iter(pool.get_new_blocks(needed))
        ledger.cache_reserved += needed
        slots = result.store_spec.block_ids.tolist()
        ids = []
        for key, slot in zip(keys, slots):
            is_borrowed = key in borrowed_blocks
            block = borrowed_blocks[key] if is_borrowed else next(allocated)
            group, index = locations[key]
            self.resident[slot] = ResidentPage(
                block, key, group, index, context, native=True, owned=True,
                reserved=not is_borrowed)
            self._resident_by_block.setdefault(block.block_id, set()).add(slot)
            pool.native_completion_block_ids.add(block.block_id)
            ids.append(block.block_id)
        result.store_spec = ResidentSlots(slots, ids, direct_gpu=True)
        return result

    def prepare_load(self, keys, req_context):
        result = super().prepare_load(keys, req_context)
        for key in keys:
            status = self._policy.get(key)
            page = self.resident.get(status.block_id) if status is not None else None
            if page is not None and page.native:
                assert status.block_id not in self.spilling
                self.memory_pool.touch([page.block])
                page.load_pins += 1
        return result

    def complete_load(self, keys, req_context):
        for key in keys:
            status = self._policy.get(key)
            page = self.resident.get(status.block_id) if status is not None else None
            if page is not None and page.native:
                assert page.load_pins > 0
                self.memory_pool.free_blocks([page.block])
                page.load_pins -= 1
        return super().complete_load(keys, req_context)

    def begin_native_reuse(self, block):
        """Called after allocator selection, before any worker may overwrite."""
        pages = []
        for slot in tuple(self._resident_by_block.get(block.block_id, ())):
            page = self.resident[slot]
            if slot in self.spilling:
                continue
            status = self._policy.get(page.key)
            assert page.native and not page.owned and not page.load_pins
            assert status is not None and status.ref_cnt == 0
            assert slot not in self.spilling and block.ref_cnt == 0
            # Pin the storage key, not the selected physical GPU block.
            super().prepare_load([page.key], page.context)
            self.spilling.add(slot)
            pages.append((slot, page))
        return pages

    def lookup(self, key, req_context):
        block = self._policy.get(key)
        if block is not None and block.block_id in self.spilling:
            return LookupResult.HIT_PENDING
        return super().lookup(key, req_context)

    def begin_spill(self, count):
        """Pin idle LRU pages until every rank has saved them to their disk slot."""
        selected = []
        for slot, page in list(self.resident.items()):
            status = self._policy.get(page.key)
            if status is None or status.ref_cnt != 0 or slot in self.spilling:
                continue
            super().prepare_load([page.key], page.context)
            self.spilling.add(slot)
            selected.append((slot, page))
            if len(selected) >= count:
                break
        return selected

    def detach_preserved_source(self, slot):
        assert slot in self.spilling
        page = self.resident.pop(slot)
        self._detached_sources[slot] = (page.key, page.context)
        assert page.native and not page.owned and not page.reserved and not page.load_pins
        slots = self._resident_by_block[page.block.block_id]
        slots.remove(slot)
        if not slots:
            del self._resident_by_block[page.block.block_id]
            self.memory_pool.native_completion_block_ids.discard(page.block.block_id)
        # Physical reference belongs to NativePressureCache, not this slot.

    def finish_spill(self, slot):
        page = self.resident.get(slot)
        key, context = (page.key, page.context) if page is not None else self._detached_sources.pop(slot)
        status = self._policy.get(key)
        assert status is not None and status.ref_cnt == 1
        self.spilling.remove(slot)
        self._release_resident(slot)
        super().complete_load([key], context)

    def discard_idle_keys(self, keys):
        for key in keys:
            block = self._policy.get(key)
            if block is not None and block.ref_cnt == 0:
                self._policy.remove(key)
                self._num_evictable_cache_blocks -= 1
                self._free_block(block)
                self._store_owners.pop(key, None)

    def prepare_store(self, keys, req_context):
        result = super().prepare_store(keys, req_context)
        if result is not None:
            for key in result.evicted_keys:
                self._store_owners.pop(key, None)
        return result

    def complete_store(self, keys, req_context, success=True):
        super().complete_store(keys, req_context, success)
        for key in keys:
            if success:
                self._store_owners[key] = req_context.req_id
            else:
                self._store_owners.pop(key, None)
            status = self._policy.get(key)
            page = self.resident.get(status.block_id) if status is not None else None
            if page is not None and page.native and page.owned:
                # Both ranks have ACKed. Normal allocator/LRU now owns lifetime;
                # the endpoint index carries no permanent GPU reservation.
                self.memory_pool.free_blocks([page.block])
                page.owned = False
                if page.reserved:
                    self.memory_ledger.cache_reserved -= 1
                    page.reserved = False

    def stored_by(self, key, request_id):
        return self._store_owners.get(key) == request_id

    def reset_cache(self):
        if self._checkpoint_pins:
            raise RuntimeError(
                "Cannot reset storage while request checkpoints are pinned"
            )
        if self.spilling:
            raise RuntimeError("Cannot reset while resident pages are spilling")
        for slot in list(self.resident):
            self._release_resident(slot)
        super().reset_cache()
        self._store_owners.clear()

    def pin_checkpoint(self, request_id, generation, keys, context):
        """Pin only a complete snapshot; incomplete candidates retain no refs.

        Keys must describe the aligned attention prefix and every recurrent
        group at the chosen boundary. The scheduler integration constructs
        that set and validates the replay distance before calling this method.
        """
        handle = (request_id, generation)
        if generation <= 0 or handle in self._checkpoint_pins:
            raise ValueError("Invalid or duplicate checkpoint generation")
        keys = tuple(dict.fromkeys(keys))
        if not keys:
            raise ValueError("Cannot pin an empty checkpoint")
        if any(self.lookup(key, context) != LookupResult.HIT for key in keys):
            return None
        slots = self.prepare_load(keys, context)
        self._checkpoint_pins[handle] = (keys, context)
        return slots

    def release_checkpoint(self, request_id, generation):
        """Release after restore/cancellation I/O has drained, never earlier."""
        keys, context = self._checkpoint_pins.pop((request_id, generation))
        self.complete_load(keys, context)

    def _get_load_store_spec(self, keys, blocks):
        slots = [block.block_id for block in blocks]
        memory = []
        for slot in slots:
            page = self.resident.get(slot)
            memory.append(page.block.block_id if page is not None else -1)
            if page is not None:
                self.resident.move_to_end(slot)
        if any(block >= 0 for block in memory):
            return ResidentSlots(slots, memory, direct_gpu=self.native_mode)
        return DiskSlots(slots)

    def get_stats(self):
        # CPU cache utilisation metrics would mislabel this disk directory.
        super().get_stats()
        if self.memory_pool is None:
            return None
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
            OffloadingConnectorStats,
        )

        stats = OffloadingConnectorStats()
        stats.set_gauge("vllm:kv_completion_resident_blocks", len(self.resident))
        stats.set_gauge("vllm:kv_completion_spilling_blocks", len(self.spilling))
        return stats


def split_transfer(gpu, disk, staging_blocks):
    """Split a 1:1 transfer without changing hybrid group or logical indices."""
    if staging_blocks <= 0 or len(gpu.block_ids) != len(disk.block_ids):
        raise ValueError("Requires positive staging and one disk slot per GPU block")
    cursor = 0
    for group, size in enumerate(gpu.group_sizes):
        for start in range(0, size, staging_blocks):
            count = min(staging_blocks, size - start)
            sizes = [0] * len(gpu.group_sizes)
            sizes[group] = count
            indices = list(gpu.block_indices)
            indices[group] += start
            sl = slice(cursor + start, cursor + start + count)
            yield (
                GPULoadStoreSpec(gpu.block_ids[sl].tolist(), sizes, indices),
                disk.block_ids[sl].tolist(),
            )
        cursor += size


class RankLocalDiskWorker(OffloadingWorker):
    def __init__(
        self,
        kv_caches,
        root,
        engine_id,
        rank,
        staging_blocks,
        page_bytes,
        verify_transfers=False,
    ):
        if staging_blocks <= 0 or page_bytes <= 0 or rank < 0:
            raise ValueError("Invalid rank-local staging geometry")
        self.device = torch.accelerator.current_device_index()
        self.root = Path(root) / engine_id / f"rank-{rank}"
        # Never interpret files from an earlier worker lifetime as valid state.
        self.root.mkdir(parents=True, exist_ok=False)
        self.staging_blocks = staging_blocks
        self.page_bytes = page_bytes
        self.rank = rank
        self.verify_transfers = verify_transfers
        self.digests = {}
        from vllm.v1.kv_offload.gb10_content_store import ContentAddressedPages
        self.content_pages = ContentAddressedPages(self.root)
        tensor_offsets = []
        offset = 0
        for tensor in kv_caches.tensors:
            tensor_offsets.append(offset)
            offset += tensor.page_size_bytes
        self.kv_caches = kv_caches
        self.tensor_offsets = tensor_offsets
        self.completion_replacements = {}
        self.completion_gpu_jobs = set()
        self.gpu_copy_stream = torch.cuda.Stream()
        self.group_spans = [
            tuple(
                sorted(
                    {(tensor_offsets[r.tensor_idx], r.page_size_bytes) for r in refs}
                )
            )
            for refs in kv_caches.group_data_refs
        ]
        self.region = SharedOffloadRegion(
            engine_id=f"{engine_id}-disk-rank-{rank}",
            num_blocks=staging_blocks,
            rank=0,
            kv_bytes_per_block=page_bytes,
            cpu_page_size=page_bytes,
        )
        try:
            self.copy = CPUOffloadingWorker(kv_caches, 1, staging_blocks, self.region)
        except Exception:
            self.region.cleanup()
            raise
        assert self.region.mmap_obj is not None
        self.view = memoryview(self.region.mmap_obj)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kv-disk")
        self.jobs = {}
        self.closed = False
        self.source_preserved = {}
        self.source_ack_ready = set()
        self.source_fence_jobs = set()
        self.eviction_pipeline = None

    def state_offset(self, state, group):
        for ref in self.kv_caches.group_data_refs[group]:
            tensor = self.kv_caches.tensors[ref.tensor_idx].tensor
            offset = state.data_ptr() - tensor.data_ptr()
            size = state[0].numel() * state.element_size()
            if offset >= 0 and offset + size <= ref.page_size_bytes:
                return self.tensor_offsets[ref.tensor_idx] + offset
        raise RuntimeError("Recurrent state missing from canonical disk page")

    def _fingerprints(self, count, group):
        result = []
        for slot in range(count):
            digest = hashlib.sha256()
            for offset, size in self.group_spans[group]:
                start = slot * self.page_bytes + offset
                digest.update(self.view[start : start + size])
            result.append(digest.digest())
        return result

    def _poison_staging(self, count, group):
        for slot in range(count):
            for offset, size in self.group_spans[group]:
                start = slot * self.page_bytes + offset
                # Bounded chunks avoid allocating a cache-page-sized Python object.
                for pos in range(start, start + size, 65536):
                    n = min(65536, start + size - pos)
                    self.view[pos : pos + n] = b"\xa5" * n

    def _submit(self, job_id, gpu, disk, store):
        if len(gpu.group_sizes) < len(self.group_spans):
            extra = len(self.group_spans) - len(gpu.group_sizes)
            gpu = GPULoadStoreSpec(
                gpu.block_ids.tolist(),
                list(gpu.group_sizes) + [0] * extra,
                list(gpu.block_indices) + [0] * extra,
            )
        if self.closed or job_id in self.jobs:
            raise RuntimeError("Closed worker or duplicate transfer ID")
        if not isinstance(disk, DiskSlots):
            raise TypeError("Rank-local transport requires DiskSlots")
        # Capture the caller's stream dependency before moving work to a thread.
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream())
        if (store and job_id in self.source_fence_jobs
                and type(disk) is DiskSlots and len(disk.block_ids) == 1
                and job_id not in self.completion_replacements):
            self._submit_staged_eviction(job_id, gpu, disk, ready)
        else:
            self.jobs[job_id] = self.executor.submit(
                self._transfer, job_id, gpu, disk, store, ready
            )
        return True

    def _init_eviction_pipeline(self):
        if self.eviction_pipeline is not None:
            return
        from vllm.v1.kv_offload.gb10_staged_evictions import StagedEvictions
        # At most 256 MiB; eight canonical pages on the current layout.
        count = max(1, min(8, (256 * 1024 * 1024) // self.page_bytes))
        region = SharedOffloadRegion(
            engine_id=f"eviction-{os.getpid()}-rank-{self.rank}",
            num_blocks=count, rank=0, kv_bytes_per_block=self.page_bytes,
            cpu_page_size=self.page_bytes)
        try:
            copy = CPUOffloadingWorker(self.kv_caches, 1, count, region)
        except Exception:
            region.cleanup()
            raise
        self.eviction_region = region
        self.eviction_copy = copy
        self.eviction_view = memoryview(region.mmap_obj)
        self.eviction_pipeline = StagedEvictions(count, self.executor)
        logger.info("GB10_ASYNC_EVICTION_STAGING rank=%d slots=%d bytes=%d",
                    self.rank, count, count * self.page_bytes)

    def _submit_staged_eviction(self, job_id, gpu, disk, ready):
        self._init_eviction_pipeline()
        started = time.monotonic()
        group = next(i for i, n in enumerate(gpu.group_sizes) if n)
        assert sum(gpu.group_sizes) == 1
        disk_slot = int(disk.block_ids[0])

        def stage(index):
            torch.accelerator.set_device_index(self.device)
            ready.synchronize()
            staging = CPULoadStoreSpec([index])
            self.eviction_copy.submit_store(job_id, gpu, staging)
            self.eviction_copy.wait({job_id})
            results = self.eviction_copy.get_finished()
            assert len(results) == 1 and results[0].success
            logger.info("GB10_EVICTION_SOURCE_PRESERVED rank=%d job=%d seconds=%.6f",
                        self.rank, job_id, time.monotonic() - started)
            return None

        def persist(index, _):
            offset = index * self.page_bytes
            digest = hashlib.sha256()
            for relative, size in self.group_spans[group]:
                digest.update(self.eviction_view[offset+relative:offset+relative+size])
            fingerprint = digest.digest()
            def write_blob(path):
                batch_store_block([path], self.eviction_view, [offset], self.page_bytes,
                                  use_o_direct=False)
            created = self.content_pages.store(disk_slot, group, fingerprint, write_blob)
            self.digests[disk_slot] = (group, fingerprint)
            elapsed = time.monotonic() - started
            size = self.page_bytes if created else 0
            logger.info("GB10_DISK_TRANSFER %s", json.dumps({
                "rank": self.rank, "job": job_id, "store": True, "bytes": size,
                "gpu_readback": self.verify_transfers, "seconds": elapsed,
                "page_bytes": self.page_bytes, "logical_bytes": self.page_bytes,
                "source_staged": True,
                "parts": [{"group": group, "slots": [disk_slot],
                           "new_payload_slots": [disk_slot] if created else []}]},
                separators=(",", ":")))
            return TransferResult(job_id, True, size, elapsed)

        preserved, completed = self.eviction_pipeline.submit(stage, persist)
        self.source_preserved[job_id] = preserved
        self.jobs[job_id] = completed

    def poll_source_preserved(self):
        # Public completions can be consumed in this same step. Save the ACK
        # before get_finished deletes its future.
        for jid, future in list(self.source_preserved.items()):
            if future.done():
                future.result()  # copy errors fail closed
                self.source_ack_ready.add(jid)
                del self.source_preserved[jid]
        result = self.source_ack_ready
        self.source_ack_ready = set()
        return result

    def wait_source_preserved(self, job_ids):
        for job_id in job_ids:
            future = self.source_preserved.get(job_id, self.jobs.get(job_id))
            if future is not None:
                future.result()

    def submit_store(self, job_id, src_spec, dst_spec):
        return self._submit(job_id, src_spec, dst_spec, True)

    def submit_load(self, job_id, src_spec, dst_spec):
        return self._submit(job_id, dst_spec, src_spec, False)

    def _transfer(self, job_id, gpu, disk, store, ready):
        torch.accelerator.set_device_index(self.device)
        ready.synchronize()
        start = time.monotonic()
        if getattr(disk, 'direct_gpu', False):
            from vllm.v1.kv_offload.gb10_gpu_completion_copy import copy_resident_pages
            if store:
                if job_id not in self.completion_gpu_jobs:
                    raise RuntimeError('Resident snapshot lacks its early GPU state capture')
                self.completion_gpu_jobs.remove(job_id)
            # All-GPU jobs bypass host staging, hashing and readback entirely.
            # Checksums/readback remain unchanged when bytes actually use disk.
            if all(b >= 0 for b in disk.memory_blocks):
                with torch.cuda.stream(self.gpu_copy_stream):
                    copy_resident_pages(self, gpu, disk, store)
                self.gpu_copy_stream.synchronize()
                logger.info('GB10_GPU_COMPLETION_TRANSFER rank=%d job=%d store=%s blocks=%d seconds=%.6f',
                            self.rank, job_id, store, len(disk.block_ids), time.monotonic()-start)
                return TransferResult(job_id, True, 0, time.monotonic()-start)
        verify_readback = self.verify_transfers
        # Experiment control: file checksum verification remains unconditional.
        # Atomic control-file replacement allows paired tests without reloading.
        try:
            control = json.loads(Path("/tmp/vllm-kv-readback-control.json").read_text())
            value = control.get("gpu_readback") if isinstance(control, dict) else None
            if type(value) is bool:
                verify_readback = value
        except (FileNotFoundError, OSError, ValueError):
            pass
        replacements = self.completion_replacements.pop(job_id, {}) if store else {}
        memory = (
            dict(zip(disk.block_ids.tolist(), disk.memory_blocks))
            if isinstance(disk, ResidentSlots)
            else {}
        )
        disk_bytes = 0
        disk_events = []
        for part, slots in split_transfer(
            gpu, disk, 1 if memory else self.staging_blocks
        ):
            group = next(i for i, n in enumerate(part.group_sizes) if n)
            resident = memory.get(slots[0], -1)
            resident_gpu = (
                GPULoadStoreSpec(
                    [resident], list(part.group_sizes), list(part.block_indices)
                )
                if resident >= 0
                else None
            )
            paths = [str(self.root / f"slot-{slot}.bin") for slot in slots]
            staging = CPULoadStoreSpec(list(range(len(slots))))
            offsets = [i * self.page_bytes for i in range(len(slots))]
            if resident_gpu is not None and getattr(disk, 'direct_gpu', False):
                from vllm.v1.kv_offload.gb10_gpu_completion_copy import copy_resident_pages
                one = ResidentSlots(slots, [resident], direct_gpu=True)
                with torch.cuda.stream(self.gpu_copy_stream):
                    copy_resident_pages(self, part, one, store)
                self.gpu_copy_stream.synchronize()
                continue
            if store:
                self.copy.submit_store(job_id, part, staging)
                self.copy.wait({job_id})
                self.copy.get_finished()
                if group in replacements:
                    assert len(slots) == 1
                    for offset, data in replacements[group]:
                        assert offset >= 0 and offset + len(data) <= self.page_bytes
                        self.view[offset : offset + len(data)] = data
                # Content fingerprints also permit exact deduplication when
                # round-trip verification is disabled in a future config.
                for slot, digest in zip(
                    slots, self._fingerprints(len(slots), group)
                ):
                    self.digests[slot] = (group, digest)
                if resident_gpu is not None:
                    self.copy.submit_load(job_id, staging, resident_gpu)
                    self.copy.wait({job_id})
                    self.copy.get_finished()
                else:
                    created = []
                    for slot, offset in zip(slots, offsets):
                        stored_group, digest = self.digests[slot]
                        assert stored_group == group
                        def write_blob(path, offset=offset):
                            batch_store_block([path], self.view, [offset], self.page_bytes,
                                              use_o_direct=False)
                        if self.content_pages.store(slot, group, digest, write_blob):
                            disk_bytes += self.page_bytes
                            created.append(slot)
                    disk_events.append({"group": group, "slots": slots,
                                        "new_payload_slots": created})
            else:
                if resident_gpu is not None:
                    self.copy.submit_store(job_id, resident_gpu, staging)
                    self.copy.wait({job_id})
                    self.copy.get_finished()
                else:
                    batch_load_block(
                        paths, self.view, offsets, self.page_bytes, use_o_direct=False
                    )
                    disk_bytes += len(slots) * self.page_bytes
                    disk_events.append({"group": group, "slots": slots})
                # Content keys always have a fingerprint. Retain verification
                # before GPU restore even when optional GPU readback is disabled.
                expected = [self.digests[slot] for slot in slots]
                actual = [(group, d) for d in self._fingerprints(len(slots), group)]
                if actual != expected:
                    raise RuntimeError("Disk bytes differ from saved GPU snapshot")
                self.copy.submit_load(job_id, staging, part)
                self.copy.wait({job_id})
                self.copy.get_finished()
                if verify_readback:
                    self._poison_staging(len(slots), group)
                    self.copy.submit_store(job_id, part, staging)
                    self.copy.wait({job_id})
                    self.copy.get_finished()
                    actual = [(group, d) for d in self._fingerprints(len(slots), group)]
                    if actual != expected:
                        raise RuntimeError(
                            "Restored GPU bytes differ from saved snapshot"
                        )
        if verify_readback and not store:
            logger.info(
                "Rank-local restore verified rank=%d job=%d blocks=%d",
                self.rank,
                job_id,
                len(disk.block_ids),
            )
        if disk_events:
            logger.info("GB10_DISK_TRANSFER %s", json.dumps({"rank": self.rank, "job": job_id,
                "store": store, "bytes": disk_bytes, "gpu_readback": verify_readback, "seconds": time.monotonic() - start,
                "page_bytes": self.page_bytes,
                "logical_bytes": sum(len(part["slots"]) for part in disk_events) * self.page_bytes,
                "parts": disk_events}, separators=(",", ":")))
        return TransferResult(
            job_id,
            True,
            disk_bytes,
            time.monotonic() - start,
        )

    def get_finished(self):
        results = []
        for job_id, future in list(self.jobs.items()):
            if future.done():
                source = self.source_preserved.get(job_id)
                if source is not None:
                    if not source.done():
                        continue
                    source.result()
                    self.source_ack_ready.add(job_id)
                # Raise on any missing/short file or copy failure. Never ACK a
                # partial transfer as successful; the engine must fail closed.
                results.append(future.result())
                del self.jobs[job_id]
                self.source_preserved.pop(job_id, None)
        return results

    def wait(self, job_ids):
        for job_id in job_ids:
            if job_id in self.source_fence_jobs:
                self.wait_source_preserved({job_id})
            elif job_id in self.jobs:
                self.jobs[job_id].result()

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        if self.eviction_pipeline is not None:
            self.eviction_pipeline.shutdown()
        self.executor.shutdown(wait=True)
        if self.eviction_pipeline is not None:
            self.eviction_view.release()
            self.eviction_copy.shutdown()
        self.view.release()
        self.copy.shutdown()


class RankLocalDiskOffloadingSpec(OffloadingSpec):
    @classmethod
    def build_metric_definitions(cls, extra_config):
        return {
            "vllm:kv_completion_resident_blocks": OffloadingGaugeMetadata(
                documentation="GPU KV pool blocks reserved by completion snapshots."
            ),
            "vllm:kv_completion_spilling_blocks": OffloadingGaugeMetadata(
                documentation="Resident completion blocks awaiting disk spill ACKs."
            ),
        }

    def __init__(self, config):
        super().__init__(config)
        if self.blocks_per_chunk != 1 or config.canonical_layout:
            raise ValueError(
                "Rank-local prototype requires direct layout, one block/chunk"
            )
        self.page_bytes = (config.worker_kv_bytes_per_block + 4095) // 4096 * 4096
        if self.page_bytes <= 0:
            raise ValueError("Cache block bytes must be positive")
        self.staging_blocks = int(self.extra_config.get("staging_blocks", 4))
        self.disk_blocks = (
            int(self.extra_config["disk_bytes_per_rank"]) // self.page_bytes
        )
        if self.staging_blocks <= 0 or self.disk_blocks <= 0:
            raise ValueError("Disk and staging budgets must hold at least one block")
        self.root = os.path.abspath(self.extra_config["root_dir"])
        self._manager = None
        self._worker = None

    def get_manager(self):
        if self._manager is None:
            self._manager = DiskSlotManager(self.disk_blocks)
        return self._manager

    def get_worker(self, kv_caches):
        if self._worker is None:
            self._worker = RankLocalDiskWorker(
                kv_caches,
                self.root,
                self.config.engine_id,
                self.config.parallel.rank,
                self.staging_blocks,
                self.page_bytes,
                verify_transfers=bool(self.extra_config.get("verify_transfers", False)),
            )
        return self._worker
