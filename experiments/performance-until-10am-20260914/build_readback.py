import json,hashlib
from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'rank_local_disk.content.py').read_text()
old='''                if self.verify_transfers:
                    expected = [self.digests[slot] for slot in slots]
                    actual = [(group, d) for d in self._fingerprints(len(slots), group)]
                    if actual != expected:
                        raise RuntimeError("Disk bytes differ from saved GPU snapshot")'''
new='''                # Content keys always have a fingerprint. Retain verification
                # before GPU restore even when optional GPU readback is disabled.
                expected = [self.digests[slot] for slot in slots]
                actual = [(group, d) for d in self._fingerprints(len(slots), group)]
                if actual != expected:
                    raise RuntimeError("Disk bytes differ from saved GPU snapshot")'''
assert old in s;s=s.replace(old,new,1)
s=s.replace('''        start = time.monotonic()
        replacements =''','''        start = time.monotonic()
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
        replacements =''',1)
s=s.replace('''                if self.verify_transfers:
                    self._poison_staging''','''                if verify_readback:
                    self._poison_staging''',1)
s=s.replace('''        if self.verify_transfers and not store:''','''        if verify_readback and not store:''',1)
s=s.replace('''"store": store, "bytes": disk_bytes,''','''"store": store, "bytes": disk_bytes, "gpu_readback": verify_readback,''',1)
f=p/'rank_local_disk.checksum_only.py';f.write_text(s);compile(s,str(f),'exec')
for rank in [0,1]:
 cfg=json.loads((p/f'candidate-content-r{rank}.json').read_text());cfg['kv_paging']['verify_transfers']=False;key='v1/kv_offload/rank_local_disk.py';cfg['runtime_overrides'][key]=str(f);cfg['runtime_override_sha256'][key]=hashlib.sha256(f.read_bytes()).hexdigest();(p/f'candidate-checksum-only-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 cfg=json.loads((p/f'candidate-content-r{rank}.json').read_text());cfg['mtp_tokens']=5;cfg['cudagraph_capture_sizes']=sorted(set(cfg['cudagraph_capture_sizes']+[160,192]));(p/f'candidate-mtp5-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
# Same real-file transport test, with readback disabled; also inject corrupt bytes.
s=(p/'content_transport_checks.py').read_text().replace('/tmp/rank_local_disk.content.py','/tmp/rank_local_disk.checksum_only.py').replace('w.verify_transfers=True','w.verify_transfers=False')
s=s.replace("  print('Transfer records',first,loaded)","""  # Corruption must fail before copying bytes to the GPU destination.
  destination_before=bytes(data[5])
  with (w.root/'slot-2.bin').open('r+b') as f:f.write(b'corrupt!')
  try:w._transfer(9,gpu([5]),m.DiskSlots([2]),False,ready)
  except RuntimeError as e:assert 'Disk bytes differ' in str(e)
  else:raise AssertionError('Corrupt cache page accepted')
  assert bytes(data[5])==destination_before
  print('Transfer records',first,loaded)""")
s=s.replace(' class Copy:\n', ' class Copy:\n  stores=0\n').replace('def submit_store(self,j,gpu,cpu):self.move(gpu,cpu,True)', 'def submit_store(self,j,gpu,cpu):self.stores+=1;self.move(gpu,cpu,True)').replace('  loaded=w._transfer', '  stores_before=w.copy.stores\n  loaded=w._transfer').replace('  assert data[3][:8]', '  assert w.copy.stores==stores_before, \'Unexpected GPU readback\'\n  assert data[3][:8]')
s=s.replace("  # Corruption must fail before copying bytes to the GPU destination.", '''  # A hot override changes only optional GPU readback, never file verification.
  count=w.copy.stores
  with patch.object(m.Path,'read_text',return_value='{"gpu_readback": true}'):
   w._transfer(90,gpu([3]),m.DiskSlots([1]),False,ready)
  assert w.copy.stores==count+1
  w.verify_transfers=True;count=w.copy.stores
  with patch.object(m.Path,'read_text',return_value='{"gpu_readback": false}'):
   w._transfer(91,gpu([3]),m.DiskSlots([1]),False,ready)
  assert w.copy.stores==count
  # Non-object or malformed controls fall back to full configured verification.
  for invalid in ('[]','null','true','{bad json', '{"gpu_readback": "false"}'):
   count=w.copy.stores
   with patch.object(m.Path,'read_text',return_value=invalid):
    w._transfer(92,gpu([3]),m.DiskSlots([1]),False,ready)
   assert w.copy.stores==count+1
  w.verify_transfers=False
  # Corruption must fail before copying bytes to the GPU destination.''')
(p/'checksum_only_checks.py').write_text(s);print('F checksum-only and G MTP5 candidates prepared; not deployed.')
