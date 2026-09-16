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

    def __init__(self, block_ids, memory_blocks):
        super().__init__(block_ids)
        self.memory_blocks = memory_blocks


@dataclass
class ResidentPage:
    block: KVCacheBlock
    key: bytes
    group: int
    index: int
    context: ReqContext


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

    def configure_memory(self, pool, ledger):
        self.memory_pool, self.memory_ledger = pool, ledger

    def _free_block(self, block):
        self._release_resident(block.block_id)
        super()._free_block(block)

    def _release_resident(self, slot):
        page = self.resident.pop(slot, None)
        if page is not None:
            assert slot not in self.spilling
            assert self.memory_pool is not None and self.memory_ledger is not None
            self.memory_pool.free_blocks([page.block])
            self.memory_ledger.cache_reserved -= 1

    def prepare_memory_store(self, keys, context, locations):
        """Allocate within the shared physical/ledger budget, before any copy."""
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

    def finish_spill(self, slot):
        page = self.resident[slot]
        status = self._policy.get(page.key)
        assert status is not None and status.ref_cnt == 1
        self.spilling.remove(slot)
        self._release_resident(slot)
        super().complete_load([page.key], page.context)

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
            return ResidentSlots(slots, memory)
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
        self.jobs[job_id] = self.executor.submit(
            self._transfer, job_id, gpu, disk, store, ready
        )
        return True

    def submit_store(self, job_id, src_spec, dst_spec):
        return self._submit(job_id, src_spec, dst_spec, True)

    def submit_load(self, job_id, src_spec, dst_spec):
        return self._submit(job_id, dst_spec, src_spec, False)

    def _transfer(self, job_id, gpu, disk, store, ready):
        torch.accelerator.set_device_index(self.device)
        ready.synchronize()
        start = time.monotonic()
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
                if self.verify_transfers:
                    expected = [self.digests[slot] for slot in slots]
                    actual = [(group, d) for d in self._fingerprints(len(slots), group)]
                    if actual != expected:
                        raise RuntimeError("Disk bytes differ from saved GPU snapshot")
                self.copy.submit_load(job_id, staging, part)
                self.copy.wait({job_id})
                self.copy.get_finished()
                if self.verify_transfers:
                    self._poison_staging(len(slots), group)
                    self.copy.submit_store(job_id, part, staging)
                    self.copy.wait({job_id})
                    self.copy.get_finished()
                    actual = [(group, d) for d in self._fingerprints(len(slots), group)]
                    if actual != expected:
                        raise RuntimeError(
                            "Restored GPU bytes differ from saved snapshot"
                        )
        if self.verify_transfers and not store:
            logger.info(
                "Rank-local restore verified rank=%d job=%d blocks=%d",
                self.rank,
                job_id,
                len(disk.block_ids),
            )
        if disk_events:
            logger.info("GB10_DISK_TRANSFER %s", json.dumps({"rank": self.rank, "job": job_id,
                "store": store, "bytes": disk_bytes, "seconds": time.monotonic() - start,
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
                # Raise on any missing/short file or copy failure. Never ACK a
                # partial transfer as successful; the engine must fail closed.
                results.append(future.result())
                del self.jobs[job_id]
        return results

    def wait(self, job_ids):
        for job_id in job_ids:
            if job_id in self.jobs:
                self.jobs[job_id].result()

    def shutdown(self):
        if self.closed:
            return
        self.closed = True
        self.executor.shutdown(wait=True)
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
