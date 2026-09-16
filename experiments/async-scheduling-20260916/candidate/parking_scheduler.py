# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parking policy shared by synchronous and version-aware asynchronous schedulers."""

import time
from typing import cast

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    gb10_aligned_offloading_connector as aligned,
)
from vllm.logger import init_logger
from vllm.v1.core.sched.parking_policy import ParkingPolicy, Phase
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec, MambaSpec

logger = init_logger(__name__)


class _ParkingMixin:
    _async_parking = False
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._async_parking:
            if (not self.scheduler_config.async_scheduling
                    or self.vllm_config.max_concurrent_batches != 2
                    or self.parallel_config.pipeline_parallel_size != 1
                    or not self.vllm_config.use_v2_model_runner):
                raise ValueError("Async parking requires V2, PP1 and two in-flight batches")
            if not getattr(self.connector, "supports_versioned_completions", False):
                raise ValueError("Async parking requires versioned completion ownership")
        elif (self.scheduler_config.async_scheduling
              or self.vllm_config.max_concurrent_batches != 1):
            raise ValueError("Synchronous parking requires single-batch scheduling")
        if (
            self.policy != SchedulingPolicy.FCFS
            or self.lora_config
            or self.is_encoder_decoder
        ):
            raise ValueError(
                "Parking prototype requires FCFS text-only serving without LoRA"
            )
        if not isinstance(self.connector, aligned.GB10AlignedOffloadingConnector):
            raise ValueError("Parking requires the aligned rank-local disk connector")
        if self.recompute_kv_load_failures:
            raise ValueError(
                "Parking must fail on load errors, never silently recompute"
            )
        transfer_config = self.vllm_config.kv_transfer_config
        assert transfer_config is not None
        opts = transfer_config.kv_connector_extra_config
        unit = int(opts.get("parking_reservation_tokens", 32768))
        if unit <= 0 or unit % 64:
            raise ValueError("Parking reservation tokens must be a positive multiple of 64")
        unit_cost, fixed_cost, recurrent_groups = 0, 0, 0
        for group in self.kv_cache_config.kv_cache_groups:
            spec = group.kv_cache_spec
            if isinstance(spec, MambaSpec):
                if spec.mamba_cache_mode != "align":
                    raise ValueError("Parking requires aligned Mamba state")
                fixed_cost += (
                    spec.max_memory_usage_bytes(self.vllm_config)
                    // spec.page_size_bytes
                )
                # Allow an additional retained checkpoint/COW endpoint.
                fixed_cost += 1
                recurrent_groups += 1
            elif isinstance(spec, CircularBufferSpec):
                fixed_cost += 1
            elif isinstance(spec, FullAttentionSpec):
                unit_cost += (unit + spec.block_size - 1) // spec.block_size
                fixed_cost += (
                    self.num_lookahead_tokens + spec.block_size - 1
                ) // spec.block_size
            else:
                raise ValueError(
                    f"Unsupported parking cache geometry: {type(spec).__name__}"
                )
        if not unit_cost or not recurrent_groups:
            raise ValueError("Expected Qwen hybrid cache geometry")
        pool = self.kv_cache_manager.block_pool
        # GPU staging is already outside the KV block pool. This headroom is
        # for retained GPU copy/offload endpoints, not CPU staging or graphs.
        physical_capacity = pool.num_gpu_blocks - 1 - (8 + 2 * recurrent_groups)
        capacity = min(
            physical_capacity, int(opts.get("parking_block_budget", physical_capacity))
        )
        self.parking = ParkingPolicy(
            capacity,
            ranks=self.parallel_config.world_size,
            unit=unit,
            unit_cost=unit_cost,
            request_cost=fixed_cost,
        )
        if self.parking.cost(self.parking.units(self.max_model_len) + 1) > capacity:
            raise ValueError(
                "Context window plus resume growth cannot fit the parking budget"
            )
        self._quiescing = {}
        self._parked = {}
        self._generations = {}
        self._park_commit = set()
        self._force_after = int(opts.get("parking_test_after_generated_tokens", 0))
        self._pressure_only = opts.get("disk_write_policy", "eager") == "pressure"
        self._memory_completions = bool(opts.get("memory_completion_cache", False))
        if self._memory_completions:
            self._parking_connector.configure_completion_memory(pool, self.parking)
        self._save_timeout = float(
            opts.get("parking_save_timeout_seconds", 300 if self._pressure_only else 30)
        )
        if not 0 < self._save_timeout < float("inf"):
            raise ValueError("parking_save_timeout_seconds must be positive and finite")
        self._forced = False
        self._forced_request = None
        logger.info(
            "GB10_PARKING budget_blocks=%d unit_tokens=%d unit_blocks=%d "
            "per_request_blocks=%d pool_blocks=%d",
            capacity,
            unit,
            unit_cost,
            fixed_cost,
            pool.num_gpu_blocks,
        )

    @property
    def _parking_connector(self) -> aligned.GB10AlignedOffloadingConnector:
        return cast(aligned.GB10AlignedOffloadingConnector, self.connector)

    def add_request(self, request):
        if request.has_encoder_inputs or request.resumable:
            raise ValueError(
                "Parking prototype supports non-streaming-input text requests only"
            )
        self.parking.enqueue(
            request.request_id, request.num_tokens + self.num_lookahead_tokens + 1
        )
        super().add_request(request)

    def _strict_waiting_order(self):
        return True

    def _can_schedule_waiting_request(self, request):
        if self._quiescing:
            return False
        rid = request.request_id
        if self._async_parking and not self._parking_connector.can_reserve_terminal(rid):
            return False
        ticket = self.parking.tickets[rid]
        if ticket.phase in (Phase.RUNNING, Phase.RESTORING):
            return (not self._async_parking or self._parking_connector.reserve_terminal(rid))
        if not self.parking.queue or self.parking.queue[0] != rid:
            return False
        admitted = self.parking.admit_head()
        if admitted is None and getattr(self, "_memory_completions", False):
            needed = self.parking.cost(
                self.parking.units(ticket.tokens) + int(ticket.phase == Phase.PARKED)
            )
            self._parking_connector.reclaim_completion_memory(needed)
        if admitted is not None:
            logger.info(
                "GB10_PARKING admit request=%s phase=%s free_blocks=%d",
                rid,
                ticket.phase.name,
                self.parking.free,
            )
        if admitted is not None and self._async_parking:
            assert self._parking_connector.reserve_terminal(rid)
        return admitted is not None

    def _select_waiting_queue_for_scheduling(self):
        # A parked/resuming request must stay ahead of unrelated external-load
        # waiters, even though vLLM normally prioritises the skipped queue.
        if self.parking.queue:
            rid = self.parking.queue[0]
            if rid in self._parked:
                request = self.requests[rid]
                for queue in (self.waiting, self.skipped_waiting):
                    if request in queue:
                        queue.remove_request(request)
                        queue.prepend_request(request)
                        return queue
        return super()._select_waiting_queue_for_scheduling()

    def _get_local_prefix_cache_hit(self, request):
        if request.request_id in self._parked:
            return self.kv_cache_manager.empty_kv_cache_blocks, 0, 0, False
        return super()._get_local_prefix_cache_hit(request)

    def _preempt_request(self, request, timestamp, drop_stale_output=False):
        if request.request_id not in self._park_commit:
            raise RuntimeError(
                "Unplanned physical KV preemption: refusing a recompute loop"
            )
        if self._async_parking:
            self._parking_connector.release_terminal(request.request_id)
        return super()._preempt_request(request, timestamp, drop_stale_output)

    def _try_park(self, request):
        rid = request.request_id
        if request.num_in_flight_tokens:
            return False
        committed = min(request.num_tokens, request.num_computed_tokens)
        generation = self._generations.get(rid, 0) + 1
        if getattr(self, "_pressure_only", False):
            started = self._parking_connector.pressure_save_started(rid)
            checkpoint = self._parking_connector.prepare_pressure_checkpoint(
                request, self.kv_cache_manager.get_block_ids(rid), generation
            )
            if not started and self._parking_connector.pressure_save_started(rid):
                self._quiescing[rid] = time.monotonic()
                logger.info(
                    "GB10_PARKING pressure_save request=%s generation=%d",
                    rid,
                    generation,
                )
        else:
            checkpoint = self._parking_connector.pin_request_checkpoint(
                request,
                committed,
                generation,
                max_replay=3200 - (request.num_tokens - committed),
            )
        if checkpoint is None:
            return False
        self._generations[rid] = generation
        self.parking.begin_save(
            rid,
            committed,
            checkpoint.boundary,
            1
            if getattr(self, "_pressure_only", False)
            else self._parking_connector._alignment,
            3200,
        )
        # A successful pin is possible only after the existing connector has
        # aggregated every rank's store ACK and published all required keys.
        for rank in range(self.parking.ranks):
            self.parking.acknowledge(rid, rank, generation, Phase.SAVING)
        self._parked[rid] = checkpoint
        self._park_commit.add(rid)
        self.running.remove(request)
        self._preempt_request(request, time.monotonic())
        self._park_commit.remove(rid)
        self._quiescing.pop(rid, None)
        if self._forced_request == rid:
            self._forced_request = None
        logger.info(
            "GB10_PARKING parked request=%s generation=%d accepted=%d "
            "boundary=%d replay=%d free_blocks=%d",
            rid,
            generation,
            request.num_tokens,
            checkpoint.boundary,
            request.num_tokens - checkpoint.boundary,
            self.parking.free,
        )
        return True

    def schedule(self, throttle_prefills=False):
        self._parking_connector.retry_completion_allocations()
        for request in list(self.running):
            rid = request.request_id
            # Rejection/draining removes speculative placeholders. Keep the
            # reservation high-water mark until park/free; the ledger's growth
            # contract is monotone even when the live token estimate shrinks.
            required_tokens = max(
                self.parking.tickets[rid].tokens,
                request.num_tokens + request.num_output_placeholders
                + self.num_lookahead_tokens + 1,
            )
            forced = (
                self._force_after > 0
                and not self._forced
                and len(self.running) >= 2
                and request.num_output_tokens >= self._force_after
                and request.num_computed_tokens >= request.num_prompt_tokens
            )
            if forced:
                self._forced = True
                self._forced_request = rid
                self._quiescing[rid] = time.monotonic()
            saving = getattr(self, "_pressure_only", False) and (
                self._parking_connector.pressure_save_started(rid)
            )
            can_grow = not saving and self.parking.grow(rid, required_tokens)
            if can_grow and rid != self._forced_request:
                if self._quiescing.pop(rid, None) is not None:
                    logger.info(
                        "GB10_PARKING growth_unblocked request=%s free_blocks=%d",
                        rid,
                        self.parking.free,
                    )
            elif not can_grow:
                self._quiescing.setdefault(rid, time.monotonic())
                if (
                    not saving
                    and getattr(self, "_memory_completions", False)
                    and rid != self._forced_request
                ):
                    needed = max(
                        0,
                        self.parking.units(required_tokens)
                        - self.parking.tickets[rid].reserved,
                    )
                    if self._parking_connector.reclaim_completion_memory(
                        needed * self.parking.unit_cost
                    ):
                        continue
            if (
                rid in self._quiescing
                and not self._try_park(request)
                and (
                    not getattr(self, "_pressure_only", False)
                    or self._parking_connector.pressure_save_started(rid)
                )
                and time.monotonic() - self._quiescing[rid]
                > getattr(self, "_save_timeout", 30)
            ):
                raise RuntimeError(f"No bounded checkpoint became available for {rid}")
        # Keep quiescing requests' GPU state and worker slots, but schedule no
        # more tokens for them while outstanding saves finish. New admissions
        # are blocked above; other running requests continue.
        held = [r for r in self.running if r.request_id in self._quiescing]
        self.running[:] = [
            r for r in self.running if r.request_id not in self._quiescing
        ]
        try:
            return super().schedule(throttle_prefills)
        finally:
            self.running.extend(held)

    def _update_waiting_for_remote_kv(self, request):
        rid = request.request_id
        checkpoint = self._parked.get(rid)
        if checkpoint is not None and rid in self.failed_recving_kv_req_ids:
            raise RuntimeError("Pinned checkpoint restore failed")
        super()._update_waiting_for_remote_kv(request)
        if checkpoint is not None:
            # Base promotion is reached only after both rank loads completed.
            for rank in range(self.parking.ranks):
                self.parking.acknowledge(
                    rid, rank, checkpoint.generation, Phase.RESTORING
                )
            self._parking_connector.release_request_checkpoint(checkpoint)
            del self._parked[rid]
            logger.info(
                "GB10_PARKING restored request=%s boundary=%d replay=%d",
                rid,
                checkpoint.boundary,
                request.num_tokens - checkpoint.boundary,
            )

    def _free_blocks(self, request):
        rid = request.request_id
        native = self._parking_connector._native_pressure
        if native is not None:
            native.deferred_requests.pop(rid, None)
        super()._free_blocks(request)
        if self._async_parking:
            self._parking_connector.release_terminal(rid)
        # Base delays this call for a cancelled request with load I/O pending.
        checkpoint = self._parked.pop(rid, None)
        if checkpoint is not None:
            self._parking_connector.release_request_checkpoint(checkpoint)
        self._quiescing.pop(rid, None)
        if self._forced_request == rid:
            self._forced_request = None
        self._generations.pop(rid, None)
        self.parking.retire(rid, io_drained=True)


class GB10ParkingScheduler(_ParkingMixin, Scheduler):
    """Original single-batch execution with the same parking policy."""


class GB10AsyncParkingScheduler(_ParkingMixin, AsyncScheduler):
    """Upstream async token accounting plus version-aware request parking."""

    _async_parking = True
