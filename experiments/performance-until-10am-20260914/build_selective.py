from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent; prior=p.parent/'aggregate-throughput-20260914'
s=(prior/'completion.aligned.py').read_text()
s=s.replace('        self.cancelled = set()','        self.cancelled = set()\n        # Only pages actually loaded into this request, never token-only matches.\n        self.restored_pages = {}')
s=s.replace('        state = s._req_status.get(rid)\n        boundary =', '        inherited = self.restored_pages.get(rid, {}) if active else self.restored_pages.pop(rid, {})\n        state = s._req_status.get(rid)\n        boundary =',1)
needle='''                key = None
                if gi < len(state.group_states) and not isinstance(spec, MambaSpec):'''
replacement='''                key = None
                # An immutable full attention page loaded from a prior snapshot
                # is already the exact state needed here. Share its existing
                # storage key while resident/ready, rather than copy it anew.
                # Partial tails, recurrent state, rings and locally shared GPU
                # pages are deliberately absent from this provenance map.
                candidate = inherited.get((gi, index))
                if candidate is not None and not isinstance(spec, (MambaSpec, CircularBufferSpec)):
                    if s.manager.lookup(candidate, state.req_context) == LookupResult.HIT:
                        key = candidate
                        shared.append(key)
                if key is None and gi < len(state.group_states) and not isinstance(spec, MambaSpec):'''
assert needle in s;s=s.replace(needle,replacement,1)
s=s.replace('''        keys, ids = [], []
        sizes =''','''        keys, ids = [], []
        restored = {}
        sizes =''',1)
needle='''            keys.append(key)
            ids.append(block.block_id)'''
replacement='''            keys.append(key)
            ids.append(block.block_id)
            spec = group_spec(self.groups[gi])
            if not isinstance(spec, (MambaSpec, CircularBufferSpec)):
                cfg = s.config.kv_group_configs[gi]
                stable_end = record.boundary // spec.block_size - int(cfg.is_eagle_group)
                if index < stable_end:
                    restored[(gi, index)] = key'''
assert needle in s;s=s.replace(needle,replacement,1)
s=s.replace('''        src = s.manager.prepare_load(keys, state.req_context)''','''        self.restored_pages[request.request_id] = restored
        src = s.manager.prepare_load(keys, state.req_context)''',1)
s=s.replace('''        record = CompletionRecord(witness_length, digest, boundary, pages)''','''        logger.info("Completion snapshot plan request=%s boundary=%d pages=%d reused=%d new=%d inherited=%d active=%s",
                    rid, boundary, len(pages), len(shared), len(sources),
                    sum(key in set(inherited.values()) for key in shared), active)
        record = CompletionRecord(witness_length, digest, boundary, pages)''',1)
(p/'completion.selective.py').write_text(s)
s=(prior/'connector.aligned.py').read_text().replace('''        old_delayed, params = super().request_finished_all_groups(request, block_ids)''','''        if self._completion is not None:
            self._completion.restored_pages.pop(request.request_id, None)
        old_delayed, params = super().request_finished_all_groups(request, block_ids)''').replace('''            self._completion.records.clear()''','''            self._completion.records.clear()
            self._completion.restored_pages.clear()''')
(p/'connector.selective.py').write_text(s)
s=(p/'rank_local_disk.baseline.py').read_text().replace('import hashlib','import hashlib\nimport json',1).replace('''        disk_bytes = 0''','''        disk_bytes = 0
        disk_events = []''',1)
s=s.replace('''                    disk_bytes += len(slots) * self.page_bytes''','''                    disk_bytes += len(slots) * self.page_bytes
                    disk_events.append({"group": group, "slots": slots})''')
s=s.replace('''        return TransferResult(
            job_id,
            True,
            disk_bytes,''','''        if disk_events:
            logger.info("GB10_DISK_TRANSFER %s", json.dumps({"rank": self.rank, "job": job_id,
                "store": store, "bytes": disk_bytes, "seconds": time.monotonic() - start,
                "page_bytes": self.page_bytes, "parts": disk_events}, separators=(",", ":")))
        return TransferResult(
            job_id,
            True,
            disk_bytes,''',1)
(p/'rank_local_disk.selective.py').write_text(s)
paths={'distributed/kv_transfer/kv_connector/v1/gb10_completion.py':'completion.selective.py','distributed/kv_transfer/kv_connector/v1/gb10_aligned_offloading_connector.py':'connector.selective.py','v1/kv_offload/rank_local_disk.py':'rank_local_disk.selective.py'}
for rank in [0,1]:
 cfg=json.loads((prior/f'candidate-aligned-r{rank}.json').read_text())
 for key,name in paths.items():
  path=p/name;compile(path.read_text(),str(path),'exec');cfg['runtime_overrides'][key]=str(path);cfg['runtime_override_sha256'][key]=hashlib.sha256(path.read_bytes()).hexdigest()
 (p/f'candidate-selective-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
print('Built selective inherited-page reuse candidate; not deployed.')
