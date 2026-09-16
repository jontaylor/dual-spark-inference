from pathlib import Path
p=Path(__file__).parent;base=p.parent/'gpu-native-completion-20260914'
s=(base/'rank_local_disk.gpu.py').read_text()
s=s.replace('        self.closed = False\n','''        self.closed = False
        self.source_preserved = {}
        self.source_fence_jobs = set()
        self.eviction_pipeline = None
''')
needle='''        self.jobs[job_id] = self.executor.submit(
            self._transfer, job_id, gpu, disk, store, ready
        )'''
s=s.replace(needle,'''        if (store and job_id in self.source_fence_jobs
                and type(disk) is DiskSlots and len(disk.block_ids) == 1
                and job_id not in self.completion_replacements):
            self._submit_staged_eviction(job_id, gpu, disk, ready)
        else:
            self.jobs[job_id] = self.executor.submit(
                self._transfer, job_id, gpu, disk, store, ready
            )''')
point='    def submit_store(self, job_id, src_spec, dst_spec):'
methods='''    def _init_eviction_pipeline(self):
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

    def wait_source_preserved(self, job_ids):
        for job_id in job_ids:
            future = self.source_preserved.get(job_id, self.jobs.get(job_id))
            if future is not None:
                future.result()

'''
s=s.replace(point,methods+point)
s=s.replace('                del self.jobs[job_id]\n','                del self.jobs[job_id]\n                self.source_preserved.pop(job_id, None)\n')
s=s.replace('        self.executor.shutdown(wait=True)\n','''        if self.eviction_pipeline is not None:
            self.eviction_pipeline.shutdown()
        self.executor.shutdown(wait=True)
        if self.eviction_pipeline is not None:
            self.eviction_view.release()
            self.eviction_copy.shutdown()
''')
(p/'rank_local_disk.async.py').write_text(s)
c=(base/'connector.gpu.py').read_text();c=c.replace('''            self.connector_worker.handle_preemptions(metadata)
            metadata.jobs_to_flush = None''','''            worker = self.connector_worker.worker
            native = set(metadata.native_eviction_jobs)
            worker.source_fence_jobs.update(native)
            try:
                self.connector_worker.handle_preemptions(metadata)
            finally:
                worker.source_fence_jobs.difference_update(native)
            metadata.jobs_to_flush = None''')
# The ordinary transport wait remains full persistence except inside this
# explicitly-scoped native source fence. No scheduler ACK changes.
s=(p/'rank_local_disk.async.py').read_text().replace('''    def wait(self, job_ids):
        for job_id in job_ids:
            if job_id in self.jobs:
                self.jobs[job_id].result()''','''    def wait(self, job_ids):
        for job_id in job_ids:
            if job_id in self.source_fence_jobs:
                self.wait_source_preserved({job_id})
            elif job_id in self.jobs:
                self.jobs[job_id].result()''')
(p/'rank_local_disk.async.py').write_text(s);(p/'connector.async.py').write_text(c)
