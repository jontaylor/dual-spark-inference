import hashlib,importlib.util,tempfile
from pathlib import Path
sp=importlib.util.spec_from_file_location('content_store',str(Path(__file__).with_name('content_store.py')));m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
with tempfile.TemporaryDirectory() as d:
 p=Path(d);c=m.ContentAddressedPages(p);writes=[]
 def save(slot,group,data):
  digest=hashlib.sha256(data).digest()
  def write(path):writes.append(path);Path(path).write_bytes(data)
  return c.store(slot,group,digest,write)
 assert save(0,0,b'first')
 assert not save(1,0,b'first') and len(writes)==1
 assert (p/'slot-0.bin').stat().st_ino==(p/'slot-1.bin').stat().st_ino
 assert not save(0,0,b'first') and (p/'slot-0.bin').read_bytes()==b'first'
 assert save(0,0,b'next') and (p/'slot-1.bin').read_bytes()==b'first'
 assert len(c.references)==2
 assert not save(1,0,b'next') and len(c.references)==1
 assert len(list(c.blobs.glob('*.bin')))==1
 assert save(2,1,b'next') and len(c.references)==2 # different group cannot alias
 before=(p/'slot-0.bin').read_bytes()
 def fail(path):Path(path).write_bytes(b'partial');raise OSError('simulated write failure')
 try:c.store(0,0,hashlib.sha256(b'fail').digest(),fail)
 except OSError:pass
 else:raise AssertionError('write failure did not propagate')
 assert (p/'slot-0.bin').read_bytes()==before and len(c.references)==2
 for i in range(100):
  save(i%3,i%2,bytes([i]))
  assert sum(c.references.values())==3
  assert len(list(c.blobs.glob('*.bin')))==len(c.references)<=3
 print('PASS: exact content reuse, group isolation, last-reference cleanup, atomic replacement, failed-write preservation, bounded slot recycling')
