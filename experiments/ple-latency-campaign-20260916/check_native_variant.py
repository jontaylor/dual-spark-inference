"""Run existing byte/error/collision tests against an experimental native library."""
import ctypes,importlib.util,pathlib,sys,tempfile
p=pathlib.Path('/home/jon/vllm-gb10-kv-paging/tests/gb10/test_ple_batch_reader.py');spec=importlib.util.spec_from_file_location('native_tests',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
lib=ctypes.CDLL(str(pathlib.Path(sys.argv[1]).resolve()));v=ctypes.c_void_p;i=ctypes.c_int64
lib.gb10_ple_reader_create.argtypes=[ctypes.c_char_p,i,i,i,ctypes.c_uint64,ctypes.POINTER(ctypes.c_int)];lib.gb10_ple_reader_create.restype=v
lib.gb10_ple_reader_gather.argtypes=[v,v,i,v,ctypes.c_uint];lib.gb10_ple_reader_destroy.argtypes=[v];lib.gb10_ple_reader_stats.argtypes=[v,v]
split=hasattr(lib,'gb10_ple_reader_submit')
if split:
 lib.gb10_ple_reader_submit.argtypes=lib.gb10_ple_reader_gather.argtypes
 lib.gb10_ple_reader_complete.argtypes=[v]
 if '--async' in sys.argv:
  lib.gb10_ple_reader_submit_async.argtypes=lib.gb10_ple_reader_gather.argtypes
  lib.gb10_ple_reader_submit=lib.gb10_ple_reader_submit_async
if '--sqpoll' in sys.argv:
 lib.gb10_ple_reader_enable_sqpoll.argtypes=[v]
 lib.gb10_ple_reader_submit_sqpoll.argtypes=lib.gb10_ple_reader_gather.argtypes
 original_create=lib.gb10_ple_reader_create
 def create_sqpoll(*args):
  handle=original_create(*args)
  if handle:
   code=lib.gb10_ple_reader_enable_sqpoll(handle)
   if code:
    assert lib.gb10_ple_reader_destroy(handle)==0
    ctypes.cast(args[-1],ctypes.POINTER(ctypes.c_int)).contents.value=code
    return None
  return handle
 lib.gb10_ple_reader_create=create_sqpoll
 lib.gb10_ple_reader_submit=lib.gb10_ple_reader_submit_sqpoll
with tempfile.TemporaryDirectory() as td:
 root=pathlib.Path(td)
 for cache in (0,1024*1024):m.test_all_reads_submitted_before_wait_and_exact_duplicate_rows(lib,root,cache)
 m.test_short_read_never_commits_partial_output(lib,root,False)
 m.test_cache_collision_does_not_overwrite_hits_before_output(lib,root)
 if split:
  m.test_short_read_never_commits_partial_output(lib,root,True)
  m.test_deferred_batch_is_not_overwritten_by_another_submission(lib,root)
 if hasattr(lib,'gb10_ple_reader_enable_sqpoll') and '--sqpoll' not in sys.argv:
  m.test_sqpoll_setup_is_optional_and_rejects_pending_batches(lib,root)
  print('PASS optional SQPOLL setup and pending ownership case')
print('PASS',6 if split else 4,'existing native byte/error/collision/pending cases')
