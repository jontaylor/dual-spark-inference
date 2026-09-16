"""Temporary-file exact-byte check for external io_uring submission tracing."""
import ctypes as c
import os
from pathlib import Path
import random
import tempfile

library = Path(__file__).resolve().parent / 'variants/17-fixed-policy/libple_batch_reader.so'
lib = c.CDLL(str(library))
lib.gb10_ple_reader_create.argtypes = [c.c_char_p, c.c_int64, c.c_int64, c.c_int64, c.c_uint64, c.POINTER(c.c_int)]
lib.gb10_ple_reader_create.restype = c.c_void_p
lib.gb10_ple_reader_enable_sqpoll.argtypes = [c.c_void_p]
lib.gb10_ple_reader_submit_sqpoll.argtypes = [c.c_void_p, c.c_void_p, c.c_int64, c.c_void_p, c.c_uint]
lib.gb10_ple_reader_complete.argtypes = [c.c_void_p]
lib.gb10_ple_reader_destroy.argtypes = [c.c_void_p]
rng = random.Random(519)
rows, width, count = 262144, 160, 9000
with tempfile.TemporaryDirectory(prefix='ple-final-order-') as directory:
    path = Path(directory) / 'table'
    data = rng.randbytes(rows * width)
    path.write_bytes(data)
    fd = os.open(path, os.O_RDONLY)
    os.fsync(fd)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    error = c.c_int()
    handle = lib.gb10_ple_reader_create(os.fsencode(path), rows, width, count, 0, c.byref(error))
    assert handle, error.value
    try:
        assert lib.gb10_ple_reader_enable_sqpoll(handle) == 0
        numbers = rng.sample(range(rows), count)
        ids = (c.c_int64 * count)(*numbers)
        output = c.create_string_buffer(count * width)
        assert lib.gb10_ple_reader_submit_sqpoll(handle, ids, count, output, 10000) == 0
        assert lib.gb10_ple_reader_complete(handle) == 0
        assert output.raw == b''.join(data[x*width:(x+1)*width] for x in numbers)
    finally:
        assert lib.gb10_ple_reader_destroy(handle) == 0
        os.close(fd)
print('PASS 9000 distinct cold-advised rows, exact bytes, explicit SQPOLL submission')
