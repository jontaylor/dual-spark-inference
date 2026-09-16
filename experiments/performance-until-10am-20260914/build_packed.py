import json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'rank_local_disk.content.py').read_text()
s=s.replace('import torch','import torch\nfrom vllm.v1.kv_offload.gb10_packed_io import store_spans, load_spans',1)
s=s.replace('''        self.region = SharedOffloadRegion(''','''        logger.info("GB10_DISK_LAYOUT %s", json.dumps({"rank": rank, "legacy_page_bytes": page_bytes,
            "group_spans": self.group_spans,
            "group_payload_bytes": [sum(size for _, size in spans) for spans in self.group_spans]}))
        self.region = SharedOffloadRegion(''',1)
s=s.replace('''            resident = memory.get(slots[0], -1)''','''            payload_size = sum(size for _, size in self.group_spans[group])
            resident = memory.get(slots[0], -1)''',1)
s=s.replace('''                            batch_store_block([path], self.view, [offset], self.page_bytes,
                                              use_o_direct=False)''','''                            store_spans(path, self.view[offset:offset + self.page_bytes],
                                        self.group_spans[group])''',1)
s=s.replace('''                            disk_bytes += self.page_bytes''','''                            disk_bytes += payload_size''',1)
s=s.replace('''                                        "new_payload_slots": created})''','''                                        "new_payload_slots": created, "payload_bytes_per_slot": payload_size})''',1)
old='''                    batch_load_block(
                        paths, self.view, offsets, self.page_bytes, use_o_direct=False
                    )
                    disk_bytes += len(slots) * self.page_bytes
                    disk_events.append({"group": group, "slots": slots})'''
new='''                    for path, offset in zip(paths, offsets):
                        load_spans(path, self.view[offset:offset + self.page_bytes],
                                   self.group_spans[group])
                    disk_bytes += len(slots) * payload_size
                    disk_events.append({"group": group, "slots": slots,
                                        "payload_bytes_per_slot": payload_size})'''
assert old in s;s=s.replace(old,new,1);f=p/'rank_local_disk.packed.py';f.write_text(s);compile(s,str(f),'exec')
for rank in [0,1]:
 cfg=json.loads((p/f'candidate-content-r{rank}.json').read_text())
 for key,name in {'v1/kv_offload/rank_local_disk.py':'rank_local_disk.packed.py','v1/kv_offload/gb10_packed_io.py':'packed_io.py'}.items():
  path=p/name;cfg['runtime_overrides'][key]=str(path);cfg['runtime_override_sha256'][key]=hashlib.sha256(path.read_bytes()).hexdigest()
 (p/f'candidate-packed-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
# Reuse transport tests, replacing only module names and adding packed helper.
s=(p/'content_transport_checks.py').read_text();marker="sp=importlib.util.spec_from_file_location('content_transport_under_test'"
i=s.index(marker)
s=s[:i]+"sp=importlib.util.spec_from_file_location('vllm.v1.kv_offload.gb10_packed_io','/tmp/packed_io.py');packed=importlib.util.module_from_spec(sp);sys.modules[sp.name]=packed;sp.loader.exec_module(packed)\n"+s[i:]
s=s.replace('/tmp/rank_local_disk.content.py','/tmp/rank_local_disk.packed.py').replace("  print('Transfer records',first,loaded)","  assert first.transfer_size==8 and loaded.transfer_size==16\n  print('Transfer records',first,loaded)")
(p/'packed_transport_checks.py').write_text(s)
print('E span-packing candidate prepared. Live group geometry/savings unknown until measured; no deployment.')
