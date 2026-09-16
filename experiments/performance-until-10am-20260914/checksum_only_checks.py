import importlib.util,sys,tempfile,types
from pathlib import Path
from unittest.mock import patch
import torch
N=types.SimpleNamespace
sp=importlib.util.spec_from_file_location('vllm.v1.kv_offload.gb10_content_store','/tmp/content_store.py');helper=importlib.util.module_from_spec(sp);sys.modules[sp.name]=helper;sp.loader.exec_module(helper)
sp=importlib.util.spec_from_file_location('content_transport_under_test','/tmp/rank_local_disk.checksum_only.py');m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
with tempfile.TemporaryDirectory() as d:
 w=m.RankLocalDiskWorker.__new__(m.RankLocalDiskWorker);w.root=Path(d);w.content_pages=helper.ContentAddressedPages(w.root);w.page_bytes=64;w.staging_blocks=2;w.group_spans=[((0,8),),((16,8),)];w.view=memoryview(bytearray(128));w.device=0;w.rank=0;w.verify_transfers=False;w.digests={};w.completion_replacements={}
 data={i:bytearray([i%256]*64) for i in [1,2,3,4,5,90]}
 data[2][:8]=data[1][:8] # equal meaningful state, different irrelevant padding
 class Copy:
  stores=0
  def move(self,gpu,cpu,store):
   group=next(i for i,n in enumerate(gpu.group_sizes) if n)
   for bid,slot in zip(gpu.block_ids,cpu.block_ids):
    for offset,size in w.group_spans[group]:
     sl=slice(int(slot)*64+offset,int(slot)*64+offset+size)
     if store:w.view[sl]=data[int(bid)][offset:offset+size]
     else:data[int(bid)][offset:offset+size]=w.view[sl]
  def submit_store(self,j,gpu,cpu):self.stores+=1;self.move(gpu,cpu,True)
  def submit_load(self,j,cpu,gpu):self.move(gpu,cpu,False)
  def wait(self,j):pass
  def get_finished(self):return []
 w.copy=Copy();ready=N(synchronize=lambda:None)
 def gpu(ids,group=0):return m.GPULoadStoreSpec(ids,[len(ids),0] if group==0 else [0,len(ids)],[0,0])
 with patch.object(torch.accelerator,'set_device_index'):
  first=w._transfer(1,gpu([1,2]),m.DiskSlots([0,1]),True,ready)
  assert (w.root/'slot-0.bin').stat().st_ino==(w.root/'slot-1.bin').stat().st_ino
  assert len(list(w.content_pages.blobs.glob('*.bin')))==1
  stores_before=w.copy.stores
  loaded=w._transfer(2,gpu([3,4]),m.DiskSlots([0,1]),False,ready)
  assert w.copy.stores==stores_before, 'Unexpected GPU readback'
  assert data[3][:8]==data[4][:8]==data[1][:8]
  # A modified meaningful span creates a different backing file; other alias survives.
  data[2][0]=77;w._transfer(3,gpu([2]),m.DiskSlots([0]),True,ready)
  assert (w.root/'slot-1.bin').read_bytes()[:8]==data[1][:8]
  # Recurrent normalization replacement bytes are fingerprinted AFTER substitution.
  w.completion_replacements[4]={0:[(0,b'newstate')]}
  w._transfer(4,gpu([2]),m.DiskSlots([2]),True,ready)
  w._transfer(5,gpu([5]),m.DiskSlots([2]),False,ready);assert data[5][:8]==b'newstate'
  # Resident save and spill retain verification and dedup correctness.
  w._transfer(6,gpu([1]),m.ResidentSlots([3],[90]),True,ready)
  assert not (w.root/'slot-3.bin').exists()
  w._transfer(7,gpu([90]),m.DiskSlots([3]),True,ready)
  assert (w.root/'slot-3.bin').stat().st_ino==(w.root/'slot-1.bin').stat().st_ino
  # Group identity is part of content identity.
  data[1][16:24]=data[1][:8];w._transfer(8,gpu([1],1),m.DiskSlots([4]),True,ready)
  assert (w.root/'slot-4.bin').stat().st_ino!=(w.root/'slot-1.bin').stat().st_ino
  # A hot override changes only optional GPU readback, never file verification.
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
  # Corruption must fail before copying bytes to the GPU destination.
  destination_before=bytes(data[5])
  with (w.root/'slot-2.bin').open('r+b') as f:f.write(b'corrupt!')
  try:w._transfer(9,gpu([5]),m.DiskSlots([2]),False,ready)
  except RuntimeError as e:assert 'Disk bytes differ' in str(e)
  else:raise AssertionError('Corrupt cache page accepted')
  assert bytes(data[5])==destination_before
  print('Transfer records',first,loaded)
 print('PASS: real filesystem transport; span-only dedup; group separation; checksum verification; normalized replacement; resident spill; immutable alias preservation')
