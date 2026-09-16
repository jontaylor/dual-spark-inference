"""One aligned control word; changes are installed on both hosts while idle."""

import ctypes
import os

MODES = {"in_process": 0, "external": 1, "abba": 2, "baab": 3}


def encode(mode, epoch, block_steps=128):
    if mode not in MODES or not 0 <= epoch < 2**32:
        raise ValueError("Invalid backend mode or epoch")
    if not 1 <= block_steps < 2**16:
        raise ValueError("block_steps must be in [1, 65535]")
    return (epoch << 32) | (block_steps << 8) | MODES[mode]


def decode(word, sequence):
    mode, block_steps, epoch = word & 255, (word >> 8) & 65535, word >> 32
    if mode not in MODES.values() or not block_steps or word & 0xFF000000:
        raise ValueError("Invalid PLE backend control word")
    if sequence < 1:
        raise ValueError("Real sequence numbers start at one")
    phase = ((sequence - 1) // block_steps) % 4
    external = mode == 1 or (mode == 2 and phase in (1, 2)) or (
        mode == 3 and phase in (0, 3)
    )
    return {
        "backend": "external" if external else "in_process",
        "epoch": epoch, "mode": mode, "block_steps": block_steps,
        "phase": phase if mode >= 2 else None,
    }


class BackendControl:
    """Read-only shared mmap; native acquire load avoids torn control reads."""

    def __init__(self, path):
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                  ctypes.c_int, ctypes.c_int, ctypes.c_int64]
        self.libc.mmap.restype = ctypes.c_void_p
        self.libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        fd = os.open(path, os.O_RDONLY)
        try:
            if os.fstat(fd).st_size != 4096:
                raise ValueError("PLE backend control must be 4096 bytes")
            self.address = self.libc.mmap(None, 4096, 1, 1, fd, 0)
        finally:
            os.close(fd)
        if self.address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), "Cannot map backend control")
        self.native = ctypes.CDLL("/opt/gb10/libmapped_wait.so")
        self.native.gb10_acquire.argtypes = [ctypes.c_void_p]
        self.native.gb10_acquire.restype = ctypes.c_uint64
        decode(self.read(), 1)

    def read(self):
        return self.native.gb10_acquire(self.address)

    def close(self):
        if self.address:
            self.libc.munmap(self.address, 4096)
            self.address = None
