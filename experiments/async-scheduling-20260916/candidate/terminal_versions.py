"""Generation-qualified ownership for terminal snapshots, independent of slots.

The worker may retire a request slot before completion-cache allocation succeeds.
Its immutable snapshot remains owned until the scheduler releases that precise
generation after the connector has drained its transfer. No live serving imports.
"""
from dataclasses import dataclass, field


class SnapshotCapacity(Exception):
    """Admission must defer; existing snapshots must not be evicted implicitly."""


@dataclass(frozen=True)
class Version:
    request_id: str
    generation: int


@dataclass
class Snapshot:
    version: Version
    slot: int
    payload: object
    active: bool = True
    transfers: set[int] = field(default_factory=set)


class TerminalVersions:
    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError('Snapshot capacity must be positive')
        self.capacity = capacity
        self.snapshots = {}
        self.slots = {}
        self._generation_highwater = {}

    def register(self, version, slot, payload):
        if version.generation <= 0 or slot < 0:
            raise ValueError('Invalid snapshot identity')
        if version in self.snapshots:
            old = self.snapshots[version]
            if old.slot != slot or old.payload is not payload:
                raise RuntimeError('Snapshot generation cannot be rebound')
            return old
        if version.generation <= self._generation_highwater.get(version.request_id, 0):
            raise RuntimeError('Snapshot generation was already retired')
        if len(self.snapshots) >= self.capacity:
            raise SnapshotCapacity()
        if slot in self.slots:
            raise RuntimeError('Active worker slot still belongs to another generation')
        snapshot = Snapshot(version, slot, payload)
        self.snapshots[version] = snapshot
        self.slots[slot] = version
        self._generation_highwater[version.request_id] = version.generation
        return snapshot

    def retire_slot(self, version):
        snapshot = self.snapshots[version]
        if snapshot.active:
            if self.slots.get(snapshot.slot) != version:
                raise RuntimeError('Worker slot ownership changed before retirement')
            del self.slots[snapshot.slot]
            snapshot.active = False

    def begin_transfer(self, version, job):
        self.snapshots[version].transfers.add(job)

    def finish_transfer(self, version, job):
        # A delayed ACK for a released generation cannot touch its successor.
        snapshot = self.snapshots.get(version)
        if snapshot is None:
            return False
        if job not in snapshot.transfers:
            return False
        snapshot.transfers.remove(job)
        return True

    def release(self, version):
        snapshot = self.snapshots.get(version)
        if snapshot is None:
            return False
        if snapshot.active or snapshot.transfers:
            raise RuntimeError('Snapshot release precedes slot retirement or I/O drain')
        del self.snapshots[version]
        return True

    def lookup(self, version):
        return self.snapshots[version].payload

    def active(self):
        return (snapshot for snapshot in self.snapshots.values() if snapshot.active)
