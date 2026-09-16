import hashlib,json
from pathlib import Path
p=Path(__file__).resolve().parent;prior=p.parent/'aggregate-throughput-20260914'
s=(p/'rank_local_disk.selective.py').read_text()
s=s.replace('''        self.digests = {}''','''        self.digests = {}
        from vllm.v1.kv_offload.gb10_content_store import ContentAddressedPages
        self.content_pages = ContentAddressedPages(self.root)''',1)
s=s.replace('''                if self.verify_transfers:
                    for slot, digest in zip(''','''                # Content fingerprints also permit exact deduplication when
                # round-trip verification is disabled in a future config.
                if True:
                    for slot, digest in zip(''',1)
old='''                    # Slots are recycled only after all readers/writers drain.
                    for path in paths:
                        Path(path).unlink(missing_ok=True)
                    batch_store_block(
                        paths, self.view, offsets, self.page_bytes, use_o_direct=False
                    )
                    disk_bytes += len(slots) * self.page_bytes
                    disk_events.append({"group": group, "slots": slots})'''
new='''                    created = []
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
                                        "new_payload_slots": created})'''
assert old in s;s=s.replace(old,new,1)
s=s.replace('''"page_bytes": self.page_bytes, "parts": disk_events''','''"page_bytes": self.page_bytes,
                "logical_bytes": sum(len(part["slots"]) for part in disk_events) * self.page_bytes,
                "parts": disk_events''',1)
# Avoid a gratuitous branch while preserving the original loop indentation.
s=s.replace('''                if True:
                    for slot, digest in zip(
                        slots, self._fingerprints(len(slots), group)
                    ):
                        self.digests[slot] = (group, digest)''','''                for slot, digest in zip(
                    slots, self._fingerprints(len(slots), group)
                ):
                    self.digests[slot] = (group, digest)''')
(p/'rank_local_disk.content.py').write_text(s)
for rank in [0,1]:
 cfg=json.loads((prior/f'candidate-aligned-r{rank}.json').read_text())
 for key,name in {'v1/kv_offload/rank_local_disk.py':'rank_local_disk.content.py','v1/kv_offload/gb10_content_store.py':'content_store.py'}.items():
  f=p/name;compile(f.read_text(),str(f),'exec');cfg['runtime_overrides'][key]=str(f);cfg['runtime_override_sha256'][key]=hashlib.sha256(f.read_bytes()).hexdigest()
 (p/f'candidate-content-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
print('Built content-dedup transport candidate. Numerical and completion sources unchanged.')
