"""Exercise snapshot ownership while allocation and transfer completion lag."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent/'candidate'))
from terminal_versions import TerminalVersions, Version, SnapshotCapacity


def raises(kind, call):
    try:
        call()
    except kind:
        return
    raise AssertionError(f'Expected {kind.__name__}')


book = TerminalVersions(2)
a, b, c = Version('same-request', 1), Version('same-request', 2), Version('C', 3)
terminal_n, next_generation = object(), object()
book.register(a, 0, terminal_n)
raises(RuntimeError, lambda: book.register(b, 0, next_generation))
book.retire_slot(a)
book.begin_transfer(a, 100)
book.register(b, 0, next_generation)
assert book.lookup(a) is terminal_n and book.lookup(b) is next_generation
raises(SnapshotCapacity, lambda: book.register(c, 1, object()))
raises(RuntimeError, lambda: book.release(a))
assert not book.finish_transfer(b, 100)
assert book.finish_transfer(a, 100)
assert not book.finish_transfer(a, 100)
assert book.release(a)
assert not book.release(a)
assert not book.finish_transfer(a, 100)
assert book.slots[0] == b and book.lookup(b) is next_generation
raises(RuntimeError, lambda: book.register(a, 1, object()))
book.register(c, 1, object())
raises(RuntimeError, lambda: book.release(b))
book.retire_slot(b)
book.retire_slot(c)
book.release(b)
book.release(c)
assert not book.snapshots and not book.slots
print(json.dumps(dict(passed=True, checks=[
    'pending snapshot survives worker-slot reuse', 'request generation isolation',
    'bounded capacity defers without eviction', 'release requires I/O drain',
    'wrong-generation and duplicate ACKs ignored', 'generation cannot be resurrected',
    'final references fully retired',
])))
