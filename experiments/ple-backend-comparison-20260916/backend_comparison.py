"""Complete legacy/existing PLE paths sharing one captured GPU output address.

Only the selected backend receives a request. Never shadow-read production
inputs. Each backend retains its own existing cache and its own input staging.
"""

import json
import os
import time
from pathlib import Path

import torch

from vllm.v1.ple_offload.backend_control import BackendControl, decode
from vllm.v1.ple_offload.connector import PleOffloadConnector
from vllm.v1.ple_offload.in_process import PleInProcessConnector


class SharedLegacyConnector(PleOffloadConnector):
    def _setup_layers(self, vllm_config, model):
        layers = super()._setup_layers(vllm_config, model)
        # Retain the actual CPU tensors across registration's storage conversion
        # and mapping. Both producers write these same allocations.
        self.output_owners = {name: layer._gpu_output_buffer
                              for name, layer in layers.items()}
        return layers


class PleBackendComparison:
    def __init__(self, vllm_config, model, device, ipc_addr, **sources):
        self.device = device
        self.rank = vllm_config.parallel_config.rank
        self.control = BackendControl(os.environ["GB10_PLE_BACKEND_CONTROL"])
        self.external = SharedLegacyConnector(
            vllm_config, model, device, ipc_addr, **sources
        )
        self.in_process = PleInProcessConnector(
            vllm_config, model, device, ipc_addr,
            shared_transport=self.external, **sources
        )
        self._done_flag = self.external._done_flag
        self._seq = 0
        self._current = None
        self._pending = []
        self._last_row = None
        self._closed = False
        root = Path(os.environ["GB10_PLE_COMPARISON_LOG_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        self.boot = f"{time.time_ns()}-{os.getpid()}"
        self.log_path = root / f"rank{self.rank}-{self.boot}.jsonl"
        self._fd = os.open(self.log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._write({"event": "init", "pid": os.getpid(),
                     "control_word": self.control.read(),
                     "cache_note": "Separate original caches; common OS file cache",
                     "output_count_note": "Sampled counts precede API stop filtering"})

    def _write(self, row):
        record = {"rank": self.rank, "boot": self.boot, **row}
        data = (json.dumps(record, separators=(",", ":")) + "\n").encode()
        if os.write(self._fd, data) != len(data):
            raise OSError("Incomplete PLE comparison telemetry write")

    @property
    def supports_deferred_completion(self):
        return (self._current is not None
                and self._current["row"]["backend"] == "in_process")

    def start_step(self):
        if self._current is not None:
            raise RuntimeError("Previous comparison step did not finish")
        self._seq += 1
        selection = decode(self.control.read(), self._seq)
        row = {"event": "step", "seq": self._seq,
               "wall_start_ns": time.time_ns(), "cpu_start_ns": time.monotonic_ns(),
               **selection}
        events = {key: torch.cuda.Event(enable_timing=True)
                  for key in ("start", "forward", "end")}
        events["start"].record()
        self._current = {"row": row, "events": events}
        self._write({**row, "event": "begin"})

    def record_batch(self, batch, full_graph):
        row = self._current["row"]
        n = batch.num_reqs
        row.update(
            requests=n, tokens=batch.num_tokens,
            padded_tokens=batch.num_tokens_after_padding,
            full_graph=bool(full_graph), has_prefill=bool(batch.has_prefill),
            scheduled_tokens=batch.num_scheduled_tokens[:n].tolist(),
            context_upper_bounds=batch.seq_lens_cpu_upper_bound[:n].tolist(),
            prefilling=batch.is_prefilling_np[:n].tolist(),
            scheduled_draft_tokens=batch.num_draft_tokens,
        )
        self._current["draft_counts"] = (
            batch.num_draft_tokens_per_req[:n].tolist()
            if batch.num_draft_tokens_per_req is not None else [0] * n
        )

    def _harvest(self):
        retained = []
        for item in self._pending:
            events, output, row = item["events"], item["output"], item["row"]
            if not events["end"].query() or not output.copy_event.query():
                retained.append(item)
                continue
            row["gpu_iteration_ms"] = events["start"].elapsed_time(events["end"])
            row["gpu_through_forward_ms"] = events["start"].elapsed_time(events["forward"])
            sampled = output.num_sampled_tokens_np[:row["requests"]].tolist()
            row["sampled_tokens_before_stop_filter"] = sum(sampled)
            row["accepted_draft_tokens"] = sum(
                min(max(int(count) - 1, 0), draft)
                for count, draft in zip(sampled, item["draft_counts"])
            )
            self._write(row)
        self._pending = retained

    def prepare_forward(self, num_reqs, num_tokens, dummy_run, *, defer_completion=False):
        if dummy_run:
            self.signal_dummy_outputs(num_tokens)
            return
        if self._current is None:
            raise RuntimeError("Missing comparison step boundary")
        row = self._current["row"]
        start = time.monotonic_ns()
        # One fence protects both producers and the captured consumer. Record
        # the preceding wait telemetry before the next producer changes flags.
        if self.external._gb10_event_recorded:
            self.external._gb10_consumed_event.synchronize()
            if self._last_row is not None:
                flags = self._done_flag
                if int(flags[24]) == self._last_row["seq"]:
                    self._last_row["gpu_wait_us"] = int(flags[17]) / 1000
                    self._last_row["gpu_wait_polls"] = int(flags[18])
                    self._last_row["gpu_wait_error"] = int(flags[16])
            self.external._gb10_event_recorded = False
            self.in_process._gb10_event_recorded = False
        row["consumer_fence_ms"] = (time.monotonic_ns() - start) / 1e6
        self._harvest()
        backend = self.external if row["backend"] == "external" else self.in_process
        backend._seq = self._seq - 1
        start = time.monotonic_ns()
        if backend is self.external:
            if defer_completion:
                raise ValueError("Legacy uses its original asynchronous worker")
            backend.prepare_forward(num_reqs, num_tokens, False)
        else:
            backend.prepare_forward(num_reqs, num_tokens, False,
                                    defer_completion=defer_completion)
        row["prepare_cpu_ms"] = (time.monotonic_ns() - start) / 1e6
        row["deferred_native"] = bool(defer_completion)
        self._last_row = row

    def finish_forward(self):
        start = time.monotonic_ns()
        self.in_process.finish_forward()
        self._current["row"]["complete_cpu_ms"] = (time.monotonic_ns() - start) / 1e6

    def release_outputs(self):
        self.external.release_outputs()
        self.in_process._gb10_event_recorded = self.external._gb10_event_recorded
        if self._current is not None:
            self._current["events"]["forward"].record()

    def end_step(self, async_output):
        if self._current is None:
            return
        self._current["events"]["end"].record()
        self._current["output"] = async_output
        self._current["row"]["cpu_enqueue_ms"] = (
            time.monotonic_ns() - self._current["row"]["cpu_start_ns"]
        ) / 1e6
        self._pending.append(self._current)
        self._current = None

    def signal_dummy_outputs(self, num_tokens):
        self.external.signal_dummy_outputs(num_tokens)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.in_process.close()
        self.external.close()
        if self._pending:
            self._pending[-1]["events"]["end"].synchronize()
            self._harvest()
        self._write({"event": "close", "last_seq": self._seq})
        os.close(self._fd)
        self.control.close()
