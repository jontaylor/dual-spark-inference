"""Paired RANDOM advice experiment on a dedicated temporary file only."""
import argparse
import ctypes as c
import json
import os
from pathlib import Path
import random
import statistics
import tempfile
import time

base = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--candidate', default='13-random-advice')
parser.add_argument('--output', default='random-advice-benchmark.json')
parser.add_argument('--idle-ms', type=float, default=0)
args = parser.parse_args()
libs = {}
for name in ('08-async-overlap', args.candidate):
    lib = c.CDLL(str(base / 'variants' / name / 'libple_batch_reader.so'))
    lib.gb10_ple_reader_create.argtypes = [c.c_char_p, c.c_int64, c.c_int64, c.c_int64, c.c_uint64, c.POINTER(c.c_int)]
    lib.gb10_ple_reader_create.restype = c.c_void_p
    for mode in ('submit', 'submit_async'):
        getattr(lib, 'gb10_ple_reader_' + mode).argtypes = [c.c_void_p, c.c_void_p, c.c_int64, c.c_void_p, c.c_uint]
    lib.gb10_ple_reader_complete.argtypes = [c.c_void_p]
    lib.gb10_ple_reader_destroy.argtypes = [c.c_void_p]
    libs[name] = lib
rng = random.Random(913)
rows, width = 262144, 160
results = []
with tempfile.TemporaryDirectory(prefix='ple-random-advice-') as directory:
    path = Path(directory) / 'table'
    data = rng.randbytes(rows * width)
    path.write_bytes(data)
    fd = os.open(path, os.O_RDONLY)
    os.fsync(fd)
    for cold in (False, True):
        for count in (64, 768, 1536):
            handles = {}
            for name, lib in libs.items():
                err = c.c_int()
                handles[name] = lib.gb10_ple_reader_create(os.fsencode(path), rows, width, count, 0, c.byref(err))
                assert handles[name], err.value
            samples = {(name, mode): [] for name in libs for mode in ('submit', 'submit_async')}
            for iteration in range(65):
                numbers = rng.sample(range(rows), count)
                ids = (c.c_int64 * count)(*numbers)
                output = c.create_string_buffer(count * width)
                expected = b''.join(data[x*width:(x+1)*width] for x in numbers)
                order = list(samples)
                rng.shuffle(order)
                for name, mode in order:
                    if cold:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    else:
                        os.pread(fd, rows * width, 0)
                    if args.idle_ms:
                        time.sleep(args.idle_ms / 1000)
                    start = time.perf_counter_ns()
                    assert getattr(libs[name], 'gb10_ple_reader_' + mode)(handles[name], ids, count, output, 10000) == 0
                    submitted = time.perf_counter_ns()
                    assert libs[name].gb10_ple_reader_complete(handles[name]) == 0
                    done = time.perf_counter_ns()
                    assert output.raw == expected
                    if iteration >= 5:
                        samples[name, mode].append(((submitted-start)/1e6, (done-start)/1e6))
            for (name, mode), values in samples.items():
                result = dict(cold_advised=cold, rows=count, variant=name, mode=mode, n=len(values))
                for index, key in enumerate(('submit_ms', 'total_ms')):
                    times = sorted(x[index] for x in values)
                    result[key] = dict(median=statistics.median(times), p95=times[int(.95*len(times))])
                results.append(result)
                print(json.dumps(result), flush=True)
            for name, handle in handles.items():
                assert libs[name].gb10_ple_reader_destroy(handle) == 0
    os.close(fd)
(base.parents[1] / 'results/ple-latency-campaign-20260916' / args.output).write_text(json.dumps(results, indent=2)+'\n')
