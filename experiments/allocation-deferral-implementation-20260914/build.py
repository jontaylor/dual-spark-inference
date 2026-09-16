"""Build isolated candidate sources; never modify live-mounted files."""
from pathlib import Path
import ast

P = Path(__file__).resolve().parent
E = P.parent


def read(path):
    return (E / path).read_text()


def replace(s, old, new, count=1):
    assert s.count(old) == count, (old[:100], s.count(old), count)
    return s.replace(old, new)


def save(name, text):
    ast.parse(text, filename=name)
    (P / name).write_text(text)


save('deferral.py', '''"""Transient capacity dependency, not an allocation failure."""
class AllocationDeferred(Exception):
    pass
''')

# Reserve physical sources before they become destinations. The pool gate is
# invoked before any multi-group allocation and also checked at get_new_blocks.
s = read('gpu-native-completion-20260914/native_pressure.gpu.py')
s = replace(s, 'from dataclasses import dataclass', 'from dataclasses import dataclass, field')
s = replace(s, '    resident_slot: int | None = None', '''    resident_slot: int | None = None
    source_block: int | None = None
    source_ranks: set = field(default_factory=set)
    source_released: bool = False''')
s = replace(s, '        pool.before_cached_block_reuse = self.before_reuse', '''        pool.before_cached_block_reuse = self.before_reuse
        pool.native_allocation_gate = self.ensure_available
        self.source_blocks = {}  # physical block -> outstanding source job IDs
        self.max_source_blocks = 64
        self.deferred_requests = {}''')
start = s.index('    def before_reuse(self, block):')
end = s.index('        if already_allocated or block.block_hash is None:', start)
s = s[:start] + '''    def ensure_available(self, required, protected=(), request_id=None):
        from vllm.v1.kv_offload.gb10_allocation_deferral import AllocationDeferred
        manager = self.scheduler.manager
        if not manager.native_mode or required <= 0:
            return
        protected = set(protected)
        # Common case: queue head already supplies the complete allocation.
        # Avoid scanning/reordering the whole cache on ordinary decode steps.
        node = self.pool.free_block_queue.fake_free_list_head.next_free_block
        ready = 0
        while node is not self.pool.free_block_queue.fake_free_list_tail and ready < required:
            if node.block_id in manager._resident_by_block and node.block_id not in protected:
                break
            ready += 1
            node = node.next_free_block
        if ready >= required:
            self.deferred_requests.pop(request_id, None)
            return
        free = self.pool.free_block_queue.get_all_free_blocks()
        # Demand includes free prefix-hit pages that will be adopted, not just
        # new destinations. Never preserve those pages for this allocation.
        usable = [b for b in free if b.block_id in protected or
                  b.block_id not in manager._resident_by_block]
        if len(usable) >= required:
            if request_id in self.deferred_requests:
                del self.deferred_requests[request_id]
            # The committed allocator still uses the native queue. Put eligible
            # pages first, preserving their relative eviction order.
            destinations = [b for b in usable if b.block_id not in protected][:required]
            for b in destinations:
                self.pool.free_block_queue.remove(b)
            self.pool.free_block_queue.prepend_n(destinations)
            return
        missing = required - len(usable)
        # Account for copies already in flight; do not repeatedly over-reserve.
        target = max(0, missing - len(self.source_blocks))
        room = max(0, self.max_source_blocks - len(self.source_blocks))
        for block in free:
            if target <= 0 or room <= 0:
                break
            if block.block_id in protected or block.block_id not in manager._resident_by_block:
                continue
            self.reserve_source(block)
            target -= 1
            room -= 1
        if not self.source_blocks:
            raise RuntimeError('Deferral shortage has no preservation progress path')
        if request_id is not None and request_id not in self.deferred_requests:
            self.deferred_requests[request_id] = time.monotonic()
            logger.info('GB10_ALLOCATION_DEFER request=%s required=%d eligible=%d sources=%d',
                        request_id, required, len(usable), len(self.source_blocks))
        raise AllocationDeferred()

    def reserve_source(self, block):
        from vllm.v1.kv_offload.rank_local_disk import DiskSlots
        assert block.ref_cnt == 0 and block.block_id not in self.source_blocks
        manager = self.scheduler.manager
        pages = manager.begin_native_reuse(block)
        assert pages
        # Remove mutable/native prefix aliases before exposing another schedule.
        self.pool._maybe_evict_cached_block(block)
        self.pool.touch([block])
        jobs = self.source_blocks[block.block_id] = set()
        for slot, page in pages:
            sizes = [0] * len(self.connector._completion_groups)
            indices = [0] * len(sizes)
            sizes[page.group], indices[page.group] = 1, page.index
            jid = self.scheduler._generate_job_id()
            jobs.add(jid)
            self.jobs[jid] = EvictionJob(
                {page.key}, self.scheduler.config.num_workers, time.monotonic(),
                slot, block.block_id)
            self.to_send[jid] = TransferJob(
                self.context.req_id,
                GPULoadStoreSpec([block.block_id], sizes, indices), DiskSlots([slot]))
            logger.info('GB10_RESERVED_EVICTION job=%d block=%d slot=%d',
                        jid, block.block_id, slot)

    def before_reuse(self, block):
        already_allocated = block.block_id in self.allocated_this_step
        self.allocated_this_step.add(block.block_id)
        manager = self.scheduler.manager
        if getattr(manager, 'native_mode', False):
            # Every caller must pass preflight. Failing closed detects missed
            # allocation paths before a worker can overwrite cached state.
            assert block.block_id not in manager._resident_by_block, \\
                'GPU-only checkpoint selected without allocation preflight'
            return
''' + s[end:]
s = replace(s, '''            meta.jobs_to_flush = (meta.jobs_to_flush or set()) | set(self.to_send)
            # The ordinary pre-forward fence is too late for request-update
            # COW/zeroing. The completion hook runs before those mutations.
            meta.native_eviction_jobs = set(self.to_send)''', '''            if self.scheduler.manager.native_mode:
                meta.reserved_eviction_jobs = set(self.to_send)
            else:
                meta.jobs_to_flush = (meta.jobs_to_flush or set()) | set(self.to_send)
                meta.native_eviction_jobs = set(self.to_send)''')
s = replace(s, '        completed = dict(meta.completed_jobs)', '''        completed = dict(meta.completed_jobs)
        for jid, ranks in getattr(meta, 'source_preserved', {}).items():
            job = self.jobs.get(jid)
            if job is None or job.source_released or job.source_block is None:
                continue  # repeated ACK cannot release another allocation
            assert job.source_block is not None
            assert set(ranks) <= set(range(self.scheduler.config.num_workers))
            job.source_ranks.update(ranks)
            if len(job.source_ranks) != self.scheduler.config.num_workers:
                continue
            self.scheduler.manager.detach_preserved_source(job.resident_slot)
            job.source_released = True
            jobs = self.source_blocks[job.source_block]
            jobs.remove(jid)
            if not jobs:
                del self.source_blocks[job.source_block]
                block = self.pool.blocks[job.source_block]
                assert block.ref_cnt == 1
                self.pool.free_blocks([block])
                logger.info('GB10_SOURCE_CAPACITY_READY block=%d job=%d', block.block_id, jid)''')
s = replace(s, '                if job.resident_slot is not None:', '''                if job.source_block is not None:
                    assert job.source_released, 'Persistence ACK preceded source ACK'
                if job.resident_slot is not None:''')
save('native_pressure.py', s)

s = read('gpu-native-completion-20260914/block_pool.gpu.py')
# Direct unguarded callers cannot accidentally pop a preservation source. They
# may raise only before allocation; normal multi-group callers preflight first.
s = replace(s, '        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)', '''        gate = getattr(self, 'native_allocation_gate', None)
        if gate is not None:
            gate(num_blocks)
        ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)''')
save('block_pool.py', s)

s = read('allocation-deferral-20260914/core__kv_cache_manager.py')
s = replace(s, '''            if required_blocks > self.block_pool.get_num_free_blocks():
                return None''', '''            if required_blocks > self.block_pool.get_num_free_blocks():
                gate = getattr(self.block_pool, 'native_allocation_gate', None)
                if gate is not None and required_blocks <= self.block_pool.get_num_free_blocks() + len(gate.__self__.source_blocks):
                    from vllm.v1.kv_offload.gb10_allocation_deferral import AllocationDeferred
                    raise AllocationDeferred()
                return None''')
s = replace(s, '''        if required_blocks > available_blocks:
            # Cannot allocate new blocks
            return None''', '''        gate = getattr(self.block_pool, 'native_allocation_gate', None)
        if gate is not None:
            protected = set()
            for manager, group in zip(self.coordinator.single_type_managers, new_computed_block_list):
                skipped = manager.get_num_skipped_tokens(num_local_computed_tokens + num_external_computed_tokens) // manager.block_size
                protected.update(b.block_id for b in group[skipped:] if not b.is_null)
            # Preserve the genuine capacity failure path when there is no copy
            # in flight to restore withheld capacity.
            owner = gate.__self__
            if required_blocks <= available_blocks + len(owner.source_blocks):
                gate(required_blocks + reserved_blocks, protected, request.request_id)
        if required_blocks > available_blocks:
            # Cannot allocate new blocks
            return None''')
save('kv_cache_manager.py', s)

s = read('allocation-deferral-20260914/core__sched__scheduler.py')
s = 'from vllm.v1.kv_offload.gb10_allocation_deferral import AllocationDeferred\n' + s
old = '''                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )'''
s = replace(s, old, '''                    try:
                        new_blocks = self.kv_cache_manager.allocate_slots(
                            request,
                            num_new_tokens,
                            num_lookahead_tokens=self.num_lookahead_tokens,
                        )
                    except AllocationDeferred:
                        new_blocks = None
                        allocation_deferred = True
                        break''')
s = replace(s, '            # Schedule newly needed KV blocks for the request.', '''            allocation_deferred = False
            # Schedule newly needed KV blocks for the request.''')
s = replace(s, '''            if new_blocks is None:
                # Cannot schedule this request.
                break''', '''            if allocation_deferred:
                req_index += 1
                continue
            if new_blocks is None:
                # Cannot schedule this request.
                break''')
start = s.index('                new_blocks = self.kv_cache_manager.allocate_slots(', s.index('                reserved_blocks = 0'))
end = s.index('\n\n                if new_blocks is None:', start)
# The running call has deeper indentation and is not this waiting call.
old = s[start:end]
s = s[:start] + '                try:\n' + '\n'.join('    '+l for l in old.splitlines()) + '''
                except AllocationDeferred:
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    if self.connector is not None and getattr(self.connector, '_completion', None):
                        self.connector._completion.release_lookup_pin(request_id)
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue''' + s[end:]
save('scheduler.py', s)

s = read('async-eviction-20260914/rank_local_disk.async.py')
s = replace(s, '    def finish_spill(self, slot):\n        page = self.resident[slot]', '''    def detach_preserved_source(self, slot):
        assert slot in self.spilling
        page = self.resident.pop(slot)
        assert page.native and not page.owned and not page.reserved and not page.load_pins
        slots = self._resident_by_block[page.block.block_id]
        slots.remove(slot)
        if not slots:
            del self._resident_by_block[page.block.block_id]
            self.memory_pool.native_completion_block_ids.discard(page.block.block_id)
        # Physical reference belongs to NativePressureCache, not this slot.

    def finish_spill(self, slot):
        page = self.resident.get(slot)''')
# finish_spill needs key/context even after GPU page metadata detached.
s = replace(s, '        self._resident_by_block = {}', '        self._resident_by_block = {}\n        self._detached_sources = {}')
s = replace(s, '        page = self.resident.pop(slot)\n        assert page.native', '        page = self.resident.pop(slot)\n        self._detached_sources[slot] = (page.key, page.context)\n        assert page.native')
s = replace(s, '''        status = self._policy.get(page.key)
        assert status is not None and status.ref_cnt == 1
        self.spilling.remove(slot)
        self._release_resident(slot)
        super().complete_load([page.key], page.context)''', '''        key, context = (page.key, page.context) if page is not None else self._detached_sources.pop(slot)
        status = self._policy.get(key)
        assert status is not None and status.ref_cnt == 1
        self.spilling.remove(slot)
        self._release_resident(slot)
        super().complete_load([key], context)''')
s = replace(s, '''        if needed > pool.get_num_free_blocks() or needed > ledger.free:
            return None
        # Borrowed pages''', '''        if needed > ledger.free:
            return None
        gate = getattr(pool, 'native_allocation_gate', None)
        if gate is not None and needed <= pool.get_num_free_blocks() + len(gate.__self__.source_blocks):
            gate(needed, request_id=context.req_id)
        if needed > pool.get_num_free_blocks():
            return None
        # Borrowed pages''')
s = replace(s, '        self.source_preserved = {}', '        self.source_preserved = {}\n        self.source_ack_ready = set()')
s = replace(s, '    def wait_source_preserved(self, job_ids):', '''    def poll_source_preserved(self):
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

    def wait_source_preserved(self, job_ids):''')
s = replace(s, '''            if future.done():
                # Raise on''', '''            if future.done():
                source = self.source_preserved.get(job_id)
                if source is not None:
                    if not source.done():
                        continue
                    source.result()
                    self.source_ack_ready.add(job_id)
                # Raise on''')
save('rank_local_disk.py', s)

s = read('allocation-deferral-20260914/distributed__kv_transfer__kv_connector__v1__offloading__common.py')
s = replace(s, '    completed_jobs: dict[int, int] = field(default_factory=dict)', '''    completed_jobs: dict[int, int] = field(default_factory=dict)
    source_preserved: dict[int, set[int]] = field(default_factory=dict)''')
s = replace(s, '        return OffloadingWorkerMetadata(\n            completed_jobs=merged,', '''        sources = {jid: set(ranks) for jid, ranks in self.source_preserved.items()}
        for jid, ranks in other.source_preserved.items():
            sources.setdefault(jid, set()).update(ranks)
        return OffloadingWorkerMetadata(
            source_preserved=sources,
            completed_jobs=merged,''')
save('common.py', s)

s = read('allocation-deferral-20260914/distributed__kv_transfer__kv_connector__v1__offloading__worker.py')
s = replace(s, '        for transfer_result in self.worker.get_finished():', '''        poll = getattr(self.worker, 'poll_source_preserved', None)
        if poll is not None:
            for jid in poll():
                self._connector_worker_meta.source_preserved[jid] = {self.worker.rank}
        for transfer_result in self.worker.get_finished():''')
s = replace(s, '        if not self._connector_worker_meta.completed_jobs:', '        if not (self._connector_worker_meta.completed_jobs or self._connector_worker_meta.source_preserved):')
s = replace(s, '        return set(), finished_recving', '''        if poll is not None:
            for jid in poll():
                self._connector_worker_meta.source_preserved[jid] = {self.worker.rank}
        return set(), finished_recving''')
save('worker.py', s)

s = read('gpu-native-completion-20260914/completion.gpu.v2.py')
s = replace(s, 'import hashlib', 'import hashlib\nimport time\nfrom vllm.v1.kv_offload.gb10_allocation_deferral import AllocationDeferred')
s = replace(s, '    native_eviction_jobs: set[int] = field(default_factory=set)', '''    native_eviction_jobs: set[int] = field(default_factory=set)
    reserved_eviction_jobs: set[int] = field(default_factory=set)''')
s = replace(s, '            completed_jobs=base.completed_jobs,', '            completed_jobs=base.completed_jobs,\n            source_preserved=base.source_preserved,')
s = replace(s, '        self.pending = {}', '        self.pending = {}\n        self.deferred_allocations = {}\n        self.skipped_allocations = set()')
s = replace(s, '''        inherited = self.restored_pages.get(rid, {}) if active else self.restored_pages.pop(rid, {})''', '''        inherited = self.restored_pages.get(rid, {})''')
s = replace(s, '''            result = s.manager.prepare_memory_store(
                list(sources), state.req_context, locations, borrowed=borrowed
            )''', '''            try:
                result = s.manager.prepare_memory_store(
                    list(sources), state.req_context, locations, borrowed=borrowed
                )
            except AllocationDeferred:
                if shared:
                    s.manager.complete_load(shared, state.req_context)
                started = self.deferred_allocations.get(rid, (None, None, None, None, time.monotonic()))[-1]
                self.deferred_allocations[rid] = (request, block_ids, active, memory, started)
                state.allocation_pending = True
                return True''')
s = replace(s, '    def add_metadata(self, meta):', '''    def retry_allocations(self):
        for rid, (request, blocks, active, memory, started) in list(self.deferred_allocations.items()):
            if time.monotonic() - started > 300:
                raise RuntimeError('Completion allocation preservation timed out')
            state = self.scheduler._req_status[rid]
            state.allocation_pending = False
            if self.finish(request, blocks, active=active, memory=memory):
                if rid in self.pending:
                    del self.deferred_allocations[rid]
                    if not active:
                        self.restored_pages.pop(rid, None)
            else:
                del self.deferred_allocations[rid]
                self.restored_pages.pop(rid, None)
                self.skipped_allocations.add(rid)
                if state.finished_signaled and not state.transfer_jobs:
                    del self.scheduler._req_status[rid]

    def add_metadata(self, meta):''')
s = replace(s, '        finished = set()', '''        finished = self.skipped_allocations
        self.skipped_allocations = set()''')
save('completion.py', s)

s = read('async-eviction-20260914/connector.async.py')
s = replace(s, '    def prepare_completion(self, runner, metadata):', '''    def retry_completion_allocations(self):
        if self._completion is not None:
            self._completion.retry_allocations()

    def prepare_completion(self, runner, metadata):
        reserved = getattr(metadata, 'reserved_eviction_jobs', None)
        if reserved:
            worker = self.connector_worker.worker
            worker.source_fence_jobs.update(reserved)
            try:
                for jid in reserved:
                    job = metadata.store_jobs.pop(jid)
                    assert worker.submit_store(jid, job.src_spec, job.dst_spec)
            finally:
                worker.source_fence_jobs.difference_update(reserved)
            reserved.clear()
            # Physical GPU sources remain reserved by the scheduler. No wait.
''')
s = replace(s, '            self._completion.restored_pages.pop(request.request_id, None)', '''            if request.request_id not in self._completion.deferred_allocations:
                self._completion.restored_pages.pop(request.request_id, None)''')
s = replace(s, '            result.transfer_stats = meta.transfer_stats', '            result.transfer_stats = meta.transfer_stats\n            result.source_preserved = meta.source_preserved')
s = replace(s, '            if self._completion.pending or self._completion.selected:', '            if self._completion.pending or self._completion.selected or self._completion.deferred_allocations:')
save('connector.py', s)

s = read('performance-until-10am-20260914/parking_scheduler.reservation.py')
s = replace(s, '    def schedule(self, throttle_prefills=False):', '''    def schedule(self, throttle_prefills=False):
        self._parking_connector.retry_completion_allocations()''')
s = replace(s, '''    def _free_blocks(self, request):
        rid = request.request_id''', '''    def _free_blocks(self, request):
        rid = request.request_id
        native = self._parking_connector._native_pressure
        if native is not None:
            native.deferred_requests.pop(rid, None)''')
save('parking_scheduler.py', s)

s = (E.parent / 'files/paging/offloading_scheduler.py').read_text()
s = replace(s, '    finished_signaled: bool = False', '''    finished_signaled: bool = False
    allocation_pending: bool = False''')
s = replace(s, '            if not req_status.transfer_jobs:\n                del self._req_status[req_id]', '''            if not req_status.transfer_jobs and not getattr(req_status, 'allocation_pending', False):
                del self._req_status[req_id]''')
s = replace(s, '            if req_status.finished_signaled and not req_status.transfer_jobs:', '''            if req_status.finished_signaled and not req_status.transfer_jobs and not getattr(req_status, 'allocation_pending', False):''')
save('offloading_scheduler.py', s)
print('Candidate sources built')
