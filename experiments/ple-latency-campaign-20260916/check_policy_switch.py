"""Exact bytes while switching native policies on one cache and reader."""
import ctypes as c
import errno
import os
from pathlib import Path
import random
import sys
import tempfile

lib = c.CDLL(str(Path(sys.argv[1]).resolve()))
lib.gb10_ple_reader_create.argtypes = [c.c_char_p, c.c_int64, c.c_int64, c.c_int64, c.c_uint64, c.POINTER(c.c_int)]
lib.gb10_ple_reader_create.restype = c.c_void_p
has_optional_sqpoll = hasattr(lib, 'gb10_ple_reader_enable_sqpoll')
if has_optional_sqpoll:
    lib.gb10_ple_reader_enable_sqpoll.argtypes = [c.c_void_p]
    lib.gb10_ple_reader_submit_async = lib.gb10_ple_reader_submit_sqpoll
for name in ('submit', 'submit_async', 'gather'):
    getattr(lib, 'gb10_ple_reader_' + name).argtypes = [c.c_void_p, c.c_void_p, c.c_int64, c.c_void_p, c.c_uint]
lib.gb10_ple_reader_complete.argtypes = [c.c_void_p]
lib.gb10_ple_reader_destroy.argtypes = [c.c_void_p]
rng = random.Random(196)
rows, width = 12000, 160
data = rng.randbytes(rows * width)
with tempfile.TemporaryDirectory(prefix='ple-switch-') as directory:
    path = Path(directory) / 'table'
    path.write_bytes(data)
    for cache in (0, 16384, 4 * 1024 * 1024):
        err = c.c_int()
        handle = lib.gb10_ple_reader_create(os.fsencode(path), rows, width, 10000, cache, c.byref(err))
        assert handle, err.value
        try:
            if has_optional_sqpoll:
                one = (c.c_int64 * 1)(0)
                untouched = c.create_string_buffer(b'X' * width)
                assert lib.gb10_ple_reader_submit_async(handle, one, 1, untouched, 10000) == -errno.EOPNOTSUPP
                assert untouched.raw[:width] == b'X' * width
                assert lib.gb10_ple_reader_enable_sqpoll(handle) == 0
                assert lib.gb10_ple_reader_enable_sqpoll(handle) == 0
            for step in range(45):
                count = (64, 768, 1536, 4097, 9000)[step % 5]
                numbers = rng.sample(range(rows), count)
                numbers[-3:] = [numbers[0]] * 3
                ids = (c.c_int64 * count)(*numbers)
                output = c.create_string_buffer(count * width)
                mode = ('submit', 'submit_async', 'gather')[step % 3]
                assert getattr(lib, 'gb10_ple_reader_' + mode)(handle, ids, count, output, 10000) == 0
                if mode != 'gather':
                    if has_optional_sqpoll:
                        assert lib.gb10_ple_reader_enable_sqpoll(handle) == -errno.EBUSY
                    assert lib.gb10_ple_reader_complete(handle) == 0
                expected = b''.join(data[x*width:(x+1)*width] for x in numbers)
                assert output.raw == expected, (cache, step, mode)
        finally:
            assert lib.gb10_ple_reader_destroy(handle) == 0
print('PASS 135 policy switches on shared reader/cache, duplicates, 64–9000 rows, three cache sizes')
