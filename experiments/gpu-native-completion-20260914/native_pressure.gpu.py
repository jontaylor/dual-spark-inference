# SPDX-License-Identifier: Apache-2.0
"""Write native cached pages only when their physical blocks are reused.

The allocator does not pin, clone, or reserve cache pages. A reuse fence sends
the old contents to storage before worker request updates can overwrite them.
Unaligned/native partial entries remain native-only; the existing aligned
connector restores complete hybrid boundaries.
"""
import time
from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import TransferJob
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id
from vllm.v1.kv_offload.base import GPULoadStoreSpec, LookupResult, ReqContext, make_offload_key

logger = init_logger(__name__)


@dataclass
class EvictionJob:
    keys: set
    pending: int
    started: float
    resident_slot: int | None = None


class NativePressureCache:
    def __init__(self, connector, pool):
        self.connector = connector
        self.scheduler = connector.connector_scheduler
        assert self.scheduler.config.blocks_per_chunk == 1
        self.context = ReqContext("gb10-native-eviction")
        self.jobs = {}
        self.to_send = {}
        # A block allocated twice before worker execution no longer contains
        # the data represented by its newly scheduled hash. Never save that hash.
        self.allocated_this_step = set()
        self.pool = pool
        pool.before_cached_block_reuse = self.before_reuse

    def before_reuse(self, block):
        already_allocated = block.block_id in self.allocated_this_step
        self.allocated_this_step.add(block.block_id)
        manager = self.scheduler.manager
        if getattr(manager, 'native_mode', False):
            if already_allocated:
                return
            from vllm.v1.kv_offload.rank_local_disk import DiskSlots
            for slot, page in manager.begin_native_reuse(block):
                sizes = [0] * len(self.connector._completion_groups)
                indices = [0] * len(sizes)
                sizes[page.group], indices[page.group] = 1, page.index
                jid = self.scheduler._generate_job_id()
                self.jobs[jid] = EvictionJob(
                    {page.key}, self.scheduler.config.num_workers, time.monotonic(), slot)
                self.to_send[jid] = TransferJob(
                    self.context.req_id,
                    GPULoadStoreSpec([block.block_id], sizes, indices), DiskSlots([slot]))
                logger.info('GB10_NATIVE_COMPLETION_EVICTION job=%d block=%d slot=%d group=%d',
                            jid, block.block_id, slot, page.group)
            return
        if already_allocated or block.block_hash is None:
            return
        assert block.ref_cnt == 0 and not block.is_null
        group = self.connector._index.get(get_group_id(block.block_hash))
        if group is None:
            return  # Compression rings have no aligned external representation.
        cfg = self.scheduler.config.kv_group_configs[group]
        boundary = block.block_hash_num_tokens
        if not boundary or boundary % cfg.tokens_per_chunk:
            return  # Never advertise partial bytes under a full-chunk key.
        key = make_offload_key(get_block_hash(block.block_hash), cfg.group_idx)
        manager = self.scheduler.manager
        if manager.lookup(key, self.context) != LookupResult.MISS:
            return
        result = manager.prepare_store([key], self.context)
        if result is None:
            # A bounded backing cache may be full of pinned parked requests.
            # Native eviction still proceeds; no false external hit is created.
            logger.info("GB10_NATIVE_EVICTION backing_full block=%d", block.block_id)
            return
        if not result.keys_to_store:
            return
        assert result.keys_to_store == [key]
        sizes = [0] * len(self.connector._completion_groups)
        indices = [0] * len(sizes)
        sizes[group] = 1
        indices[group] = boundary // cfg.tokens_per_chunk - 1
        jid = self.scheduler._generate_job_id()
        self.jobs[jid] = EvictionJob({key}, self.scheduler.config.num_workers, time.monotonic())
        self.to_send[jid] = TransferJob(
            self.context.req_id,
            GPULoadStoreSpec([block.block_id], sizes, indices),
            result.store_spec,
        )
        logger.info("GB10_NATIVE_EVICTION save job=%d block=%d group=%d boundary=%d",
                    jid, block.block_id, group, boundary)

    def add_metadata(self, meta):
        if self.to_send:
            meta.store_jobs.update(self.to_send)
            meta.jobs_to_flush = (meta.jobs_to_flush or set()) | set(self.to_send)
            # The ordinary pre-forward fence is too late for request-update
            # COW/zeroing. The completion hook runs before those mutations.
            meta.native_eviction_jobs = set(self.to_send)
            self.to_send.clear()
        self.allocated_this_step.clear()
        if any(time.monotonic() - job.started > 300 for job in self.jobs.values()):
            raise RuntimeError("Native prefix eviction save timed out")

    def consume_completions(self, meta):
        completed = dict(meta.completed_jobs)
        for jid, job in list(self.jobs.items()):
            job.pending -= completed.pop(jid, 0)
            assert job.pending >= 0
            if not job.pending:
                if job.resident_slot is not None:
                    self.scheduler.manager.finish_spill(job.resident_slot)
                else:
                    self.scheduler.manager.complete_store(job.keys, self.context)
                del self.jobs[jid]
        return completed
