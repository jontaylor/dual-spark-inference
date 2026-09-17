# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission/resume ledger for same-process paging.

This policy has no I/O and does not free cache blocks. The integration must
quiesce workers before begin_save and acknowledge each rank only after its
snapshot is committed. Capacity is in credits, with configurable costs per
token unit and per request, net of staging and allocator headroom. The opt-in
GB10ParkingScheduler supplies the physical block costs and I/O integration.
"""

from dataclasses import dataclass, field
from enum import Enum, auto


class Phase(Enum):
    QUEUED = auto()
    RUNNING = auto()
    SAVING = auto()
    PARKED = auto()
    RESTORING = auto()
    FAILED = auto()


@dataclass
class Ticket:
    tokens: int
    phase: Phase = Phase.QUEUED
    reserved: int = 0
    checkpoint: int | None = None
    generation: int = 0
    acknowledgements: set[int] = field(default_factory=set)


class ParkingPolicy:
    def __init__(
        self,
        capacity: int,
        ranks: int = 2,
        unit: int = 32768,
        unit_cost: int = 1,
        request_cost: int = 0,
    ):
        if min(capacity, ranks, unit, unit_cost) <= 0 or request_cost < 0:
            raise ValueError("Capacity, ranks and unit must be positive")
        self.capacity, self.ranks, self.unit = capacity, ranks, unit
        self.unit_cost, self.request_cost = unit_cost, request_cost
        self.tickets: dict[str, Ticket] = {}
        self.queue: list[str] = []
        self.cache_reserved = 0

    @property
    def free(self) -> int:
        return (
            self.capacity
            - self.cache_reserved
            - sum(self.cost(t.reserved) for t in self.tickets.values())
        )

    def cost(self, units: int) -> int:
        return units * self.unit_cost + (self.request_cost if units else 0)

    def units(self, tokens: int) -> int:
        if tokens < 0:
            raise ValueError("Negative token footprint")
        return max(1, (tokens + self.unit - 1) // self.unit)

    def enqueue(self, request_id: str, tokens: int = 0) -> None:
        if request_id in self.tickets:
            raise ValueError("Duplicate request")
        if self.cost(self.units(tokens)) > self.capacity:
            raise ValueError("Request cannot fit alone")
        self.tickets[request_id] = Ticket(tokens)
        self.queue.append(request_id)

    def admit_head(self) -> str | None:
        if not self.queue:
            return None
        rid = self.queue[0]
        t = self.tickets[rid]
        if t.phase not in (Phase.QUEUED, Phase.PARKED):
            return None
        # Saved tokens must fit together with the next growth unit. The
        # integration must reject contexts for which this is impossible.
        needed = self.units(t.tokens) + int(t.phase == Phase.PARKED)
        if self.cost(needed) > self.capacity:
            raise ValueError("Resume plus growth cannot fit alone")
        if self.cost(needed) > self.free:
            return None
        t.reserved = needed
        if t.phase == Phase.PARKED:
            t.phase = Phase.RESTORING
            t.acknowledgements.clear()
            # Keep the queue barrier while I/O is in flight.
        else:
            t.phase = Phase.RUNNING
            self.queue.pop(0)
        return rid

    def grow(self, request_id: str, tokens: int) -> bool:
        t = self.tickets[request_id]
        if t.phase != Phase.RUNNING or tokens < t.tokens:
            raise ValueError("Only running requests may grow monotonically")
        needed = self.units(tokens)
        extra = max(0, needed - t.reserved)
        if extra * self.unit_cost > self.free:
            return False
        t.tokens = tokens
        t.reserved += extra
        return True

    def begin_save(
        self,
        request_id: str,
        committed: int,
        checkpoint: int,
        alignment: int,
        max_replay: int,
    ) -> int:
        t = self.tickets[request_id]
        if t.phase != Phase.RUNNING:
            raise ValueError("Quiesce a running request before saving")
        if (
            alignment <= 0
            or max_replay < 0
            or checkpoint <= 0
            or checkpoint % alignment
            or committed < checkpoint
            or committed > t.tokens
            or committed - checkpoint > max_replay
        ):
            raise ValueError("No compatible checkpoint within replay bound")
        if self.cost(self.units(committed) + 1) > self.capacity:
            raise ValueError("Resume plus growth cannot fit alone")
        t.tokens, t.checkpoint = committed, checkpoint
        t.phase = Phase.SAVING
        t.generation += 1
        t.acknowledgements.clear()
        self.queue.insert(0, request_id)
        return t.generation

    def acknowledge(
        self, request_id: str, rank: int, generation: int, phase: Phase
    ) -> bool:
        t = self.tickets[request_id]
        if not 0 <= rank < self.ranks:
            raise ValueError("Invalid rank")
        if (
            generation != t.generation
            or phase != t.phase
            or phase not in (Phase.SAVING, Phase.RESTORING)
        ):
            return False  # stale completion must never free another snapshot
        t.acknowledgements.add(rank)
        if len(t.acknowledgements) != self.ranks:
            return False
        if phase == Phase.SAVING:
            t.phase, t.reserved = Phase.PARKED, 0
        else:
            t.phase = Phase.RUNNING
            self.queue.remove(request_id)
        return True

    def fail_transfer(self, request_id: str) -> None:
        t = self.tickets[request_id]
        if t.phase not in (Phase.SAVING, Phase.RESTORING):
            raise ValueError("No transfer in progress")
        # Do not free memory or unblock admissions on an I/O error. The
        # integration cancels/drains outstanding operations before retire().
        t.phase = Phase.FAILED

    def retire(self, request_id: str, *, io_drained: bool = False) -> None:
        t = self.tickets[request_id]
        if t.phase in (Phase.SAVING, Phase.RESTORING, Phase.FAILED) and not io_drained:
            raise ValueError("Drain transfers before releasing reservations")
        if request_id in self.queue:
            self.queue.remove(request_id)
        del self.tickets[request_id]
