"""Bounded asynchronous CUDA-event stage timing. Never synchronizes inference.

Enabled by /tmp/spec-cost-control.json; output contains timing and batch sizes only.
Events are recorded outside captured graphs. Samples include main-stream waits and
host submission gaps, not just kernel busy time. Other streams count only where
execution already establishes a dependency on them.
"""
import json
import pathlib
import time
from collections import deque
import torch


class SpecCostProbe:
    def __init__(self):
        self.control_path = pathlib.Path('/tmp/spec-cost-control.json')
        self.rank = None
        self.control = {}
        self.phase = None
        self.last_poll = 0.0
        self.seen = 0
        self.selected = 0
        self.active = None
        self.pending = deque()
        self.pool = []
        self.disabled_error = False
        self.last_real_start = None
        self.origin = None
        self.nvtx_open = False

    def _emit(self, value):
        with pathlib.Path(f'/tmp/spec-cost-rank{self.rank}.jsonl').open('a') as f:
            f.write(json.dumps(value) + '\n')

    def _fail(self, exc):
        if self.nvtx_open:
            torch.cuda.nvtx.range_pop()
            self.nvtx_open = False
        self.disabled_error = True
        self.active = None
        try:
            self._emit({'type': 'probe_error', 'time': time.time(), 'error': repr(exc)})
        except Exception:
            pass

    def _drain(self):
        while self.pending and self.pending[0]['events'][self.pending[0]['used'] - 1].query():
            r = self.pending.popleft()
            events = r.pop('events')
            used = r.pop('used')
            r['gpu_start_ms'] = self.origin.elapsed_time(events[0])
            r['gpu_elapsed_ms'] = {name: events[0].elapsed_time(events[i]) for i, name in enumerate(r['marks'])}
            stamps = r.pop('cpu_stamps')
            r['cpu_elapsed_ms'] = {name: (stamp - stamps[0]) * 1000 for name, stamp in zip(r['marks'], stamps)}
            self._emit(r)
            self.pool.append(events)

    def begin(self):
        if self.disabled_error:
            return
        try:
            if self.rank is None:
                self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            self._drain()
            now = time.time()
            previous_start = self.last_real_start
            self.last_real_start = time.perf_counter()
            if self.active is not None:
                self.active['abandoned'] = True
                self.end()
            if now - self.last_poll > 0.5:
                self.last_poll = now
                try:
                    self.control = json.loads(self.control_path.read_text())
                except (FileNotFoundError, json.JSONDecodeError):
                    self.control = {}
            c = self.control
            if not c.get('enabled') or now > c.get('until', 0):
                return
            phase = str(c.get('phase', 'unnamed'))
            if phase != self.phase:
                self.phase = phase
                self.seen = 0
                self.selected = 0
                self._emit({'type': 'phase_start', 'phase': phase, 'rank': self.rank, 'time': now})
            self.seen += 1
            stride = max(1, int(c.get('stride', 4)))
            if self.selected >= min(1000, int(c.get('limit', 300))) or (self.seen - 1) % stride:
                return
            if not self.pool and len(self.pending) >= 32:
                return
            if self.origin is None:
                self.origin = torch.cuda.Event(enable_timing=True)
                self.origin.record()
            events = self.pool.pop() if self.pool else [torch.cuda.Event(enable_timing=True) for _ in range(24)]
            self.selected += 1
            self.active = {'type': 'sample', 'rank': self.rank, 'phase': phase, 'sample': self.selected, 'step': self.seen, 'time': now, 'stride': stride, 'warmup': self.selected <= int(c.get('warmup', 10)), 'events': events, 'used': 0, 'marks': [], 'cpu_stamps': [], 'prior_step_start_gap_ms': None if previous_start is None else (self.last_real_start - previous_start)*1000}
            self.mark('cycle_start')
        except Exception as exc:
            self._fail(exc)

    def batch(self, batch, graph_mode):
        if self.active is None:
            return
        try:
            lens = batch.seq_lens_cpu_upper_bound
            assert lens.device.type == 'cpu'
            self.active.update(num_reqs=int(batch.num_reqs), num_tokens=int(batch.num_tokens), num_tokens_padded=int(batch.num_tokens_after_padding), num_draft_tokens=int(batch.num_draft_tokens), has_prefill=bool(batch.has_prefill), graph_mode=str(graph_mode), scheduled_tokens=batch.num_scheduled_tokens.tolist(), context_lengths=lens[:batch.num_reqs].tolist())
        except Exception as exc:
            self._fail(exc)

    def mark(self, name):
        if self.active is None:
            return
        try:
            r = self.active
            if self.control.get('nvtx'):
                if self.nvtx_open:
                    torch.cuda.nvtx.range_pop()
                    self.nvtx_open = False
                if name != 'cycle_end':
                    torch.cuda.nvtx.range_push('spec-cost/' + name)
                    self.nvtx_open = True
            n = r['used']
            if n >= len(r['events']):
                raise RuntimeError('Probe event budget exhausted')
            r['events'][n].record()
            r['marks'].append(name)
            r['cpu_stamps'].append(time.perf_counter())
            r['used'] += 1
        except Exception as exc:
            self._fail(exc)

    def end(self):
        if self.active is None:
            return
        self.mark('cycle_end')
        if self.active is not None:
            self.pending.append(self.active)
            self.active = None
