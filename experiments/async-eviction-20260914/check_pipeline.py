import threading
from concurrent.futures import ThreadPoolExecutor
from pipeline import StagedEvictions

def require(event):assert event.wait(3), 'timed out'
with ThreadPoolExecutor(max_workers=1) as io:
 p=StagedEvictions(1,io);gate=threading.Event();writing=threading.Event();staged2=threading.Event()
 def persist(slot,data):writing.set();require(gate);return data
 first,done=p.submit(lambda slot:b'old GPU bytes',persist);first.result(3);require(writing)
 assert not done.done(), 'persistence must not ACK at source fence'
 second,done2=p.submit(lambda slot:staged2.set() or b'next',lambda slot,data:data)
 assert not staged2.wait(.1), 'bounded staging must backpressure'
 gate.set();assert done.result(3)==b'old GPU bytes';second.result(3);assert done2.result(3)==b'next'
 def fail(_):raise ValueError('copy failed')
 f,d=p.submit(fail,lambda *_:None)
 for x in [f,d]:
  try:x.result(3);raise AssertionError('copy failure lost')
  except ValueError:pass
 def fail_write(*_):raise OSError('disk failed')
 f,d=p.submit(lambda _:b'bytes',fail_write);f.result(3)
 try:d.result(3);raise AssertionError('write failure ACKed')
 except OSError:pass
 f,d=p.submit(lambda _:b'after failure',lambda _,x:x);f.result(3);assert d.result(3)==b'after failure'
 p.shutdown()
print('{"passed":true,"checks":["source fence before disk ACK","bounded backpressure","copy failure propagation","disk failure propagation","buffer recovery"]}')
