"""Exact span I/O for worker-lifetime cache files; staging padding is excluded."""
import os


def _views(page, spans):
    result = []
    for offset, size in spans:
        if offset < 0 or size < 0 or offset + size > len(page):
            raise ValueError('Cache span outside staging page')
        if size:
            result.append(page[offset:offset + size])
    if not result:
        raise ValueError('Empty cache payload')
    return result


def _all(fd, views, operation):
    # Short vectored I/O can stop within a span. Keep advancing until complete.
    limit = int(os.sysconf('SC_IOV_MAX'))
    views = list(views)
    while views:
        try:
            count = operation(fd, views[:limit])
        except InterruptedError:
            continue
        if count <= 0:
            raise OSError('Incomplete cache payload I/O')
        while views and count >= len(views[0]):
            count -= len(views.pop(0))
        if count:
            views[0] = views[0][count:]


def store_spans(path, page, spans):
    views = _views(page, spans)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _all(fd, views, os.writev)
    finally:
        os.close(fd)


def load_spans(path, page, spans):
    views = _views(page, spans)
    expected = sum(map(len, views))
    fd = os.open(path, os.O_RDONLY)
    try:
        if os.fstat(fd).st_size != expected:
            raise OSError('Cache payload size does not match group layout')
        _all(fd, views, os.readv)
    finally:
        os.close(fd)
