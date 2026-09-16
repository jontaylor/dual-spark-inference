import tempfile,os
from pathlib import Path
from unittest.mock import patch
from packed_io import store_spans,load_spans
with tempfile.TemporaryDirectory() as directory:
 path=Path(directory)/'page';original=bytearray(range(128));source=memoryview(original);spans=[(3,11),(48,24),(95,17)]
 # Force repeated short syscalls that stop both inside spans and between them.
 writev=os.writev;readv=os.readv
 def shortwrite(fd,views):return writev(fd,[views[0][:5]])
 def shortread(fd,views):return readv(fd,[views[0][:3]])
 with patch('os.writev',shortwrite):store_spans(path,source,spans)
 assert path.stat().st_size==52
 destination=bytearray([255]*128)
 with patch('os.readv',shortread):load_spans(path,memoryview(destination),spans)
 covered={i for a,n in spans for i in range(a,a+n)}
 assert all(destination[i]==(original[i] if i in covered else 255) for i in range(128))
 for data in [b'short',path.read_bytes()+b'extra']:
  path.write_bytes(data)
  try:load_spans(path,memoryview(destination),spans)
  except OSError:pass
  else:raise AssertionError('Malformed payload accepted')
 try:store_spans(Path(directory)/'bad',source,[(120,20)])
 except ValueError:pass
 else:raise AssertionError('Out-of-page span accepted')
 # Full-page layouts are valid and gain no packing reduction.
 full=Path(directory)/'full';store_spans(full,source,[(0,128)]);assert full.read_bytes()==original
 print('PASS: short vector writes/reads, exact active bytes, untouched padding, malformed length rejection, span bounds, full-page compatibility')
