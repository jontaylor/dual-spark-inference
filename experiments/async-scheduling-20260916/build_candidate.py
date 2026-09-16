"""Assemble an isolated async candidate. Never edits deployment or serving files."""
import json
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
OUT = P/'candidate'
CFG = json.loads((ROOT/'deploy_config.json').read_text())


def source(key):
    return Path(CFG['runtime_overrides'][key]).read_text()


def once(text, old, new):
    assert text.count(old) == 1, (old[:90], text.count(old))
    return text.replace(old, new)


completion = source('distributed/kv_transfer/kv_connector/v1/gb10_completion.py')
completion = once(completion, '    job_id: int\n',
                  '    job_id: int\n    terminal_version: int | None = None\n    request_id: str = ""\n')
completion = once(completion, 'class CompletionMetadata(OffloadingConnectorMetadata):\n',
    'class CompletionMetadata(OffloadingConnectorMetadata):\n'
    '    terminal_versions: dict[str, int] = field(default_factory=dict, kw_only=True)\n'
    '    terminal_releases: list[tuple[str, int]] = field(default_factory=list, kw_only=True)\n')
completion = once(completion, '        inherited = self.restored_pages.get(rid, {})\n',
    '        inherited = self.restored_pages.get(rid, {})\n'
    '        terminal_version = (self.connector.terminal_version(rid)\n'
    '                            if not active and self.connector.supports_versioned_completions else None)\n')
completion = once(completion, '            or request.num_in_flight_tokens\n',
    '            or (request.num_in_flight_tokens and terminal_version is None)\n')
completion = once(completion, '                        and not isinstance(spec, MambaSpec)\n',
    '                        and not isinstance(spec, MambaSpec)\n'
    '                        and not (terminal_version is not None and isinstance(spec, CircularBufferSpec))\n')
completion = once(completion, "                if key in sources and not isinstance(group_spec(self.groups[gi]), MambaSpec)\n",
    "                if key in sources and not isinstance(group_spec(self.groups[gi]), MambaSpec)\n"
    "                and not (terminal_version is not None and isinstance(group_spec(self.groups[gi]), CircularBufferSpec))\n")
completion = once(completion, '        save = CompletionSave(record, block_ids, jid)\n',
    '        save = CompletionSave(record, block_ids, jid, terminal_version, rid)\n')
needle = '    for rid, save in metadata.completion_saves.items():\n'
completion = once(completion, needle, needle+
    '        if save.terminal_version is not None:\n'
    '            replacements = runner._terminal_worker.disk_replacements(save)\n'
    '            if replacements is None:\n'
    '                connector._invalid_completions.add(save.job_id)\n'
    '            else:\n'
    '                worker.completion_replacements[save.job_id] = replacements\n'
    '            continue\n')
(OUT/'completion.py').write_text(completion)

connector = source('distributed/kv_transfer/kv_connector/v1/gb10_aligned_offloading_connector.py')
connector = once(connector, '        self._pinned_checkpoints = {}\n',
    '        self.supports_versioned_completions = bool(vllm_config.scheduler_config.async_scheduling)\n'
    '        self._terminal_versions = {}\n'
    '        self._terminal_releases = []\n'
    '        self._terminal_generation = 0\n'
    '        self._terminal_capacity = vllm_config.scheduler_config.max_num_seqs\n'
    '        self._pinned_checkpoints = {}\n')
connector = once(connector, '            or vllm_config.scheduler_config.async_scheduling\n', '')
connector = once(connector, '        self.native_completions = bool(extra.get(\'native_completion_cache\', False))\n',
    '        self.native_completions = bool(extra.get(\'native_completion_cache\', False))\n'
    '        if self.supports_versioned_completions and (\n'
    '                not self.native_completions or not self.memory_completions\n'
    '                or not self.pressure_only or int(extra.get("async_terminal_snapshot_bytes", 0)) <= 0):\n'
    '            raise ValueError("Async paging requires native completion versions and a memory budget")\n')
connector = once(connector, '        self._spill_to_send.clear()\n',
    '        self._spill_to_send.clear()\n'
    '        if self.supports_versioned_completions:\n'
    '            meta.terminal_versions = {r.req_id: self.terminal_version(r.req_id)\n'
    '                                      for r in scheduler_output.scheduled_new_reqs}\n'
    '            meta.terminal_releases = self._terminal_releases\n'
    '            self._terminal_releases = []\n')
connector += '''

    def can_reserve_terminal(self, rid):
        return rid in self._terminal_versions or len(self._terminal_versions) < self._terminal_capacity

    def reserve_terminal(self, rid):
        if rid not in self._terminal_versions:
            if not self.can_reserve_terminal(rid):
                return False
            self._terminal_generation += 1
            self._terminal_versions[rid] = self._terminal_generation
        return True

    def terminal_version(self, rid):
        return self._terminal_versions[rid]

    def release_terminal(self, rid):
        generation = self._terminal_versions.pop(rid, None)
        if generation is not None:
            self._terminal_releases.append((rid, generation))
'''
(OUT/'connector.py').write_text(connector)

gpu_copy = source('v1/kv_offload/gb10_gpu_completion_copy.py')
gpu_copy = once(gpu_copy, '        worker.completion_gpu_jobs.add(save.job_id)\n',
    '        worker.completion_gpu_jobs.add(save.job_id)\n'
    '        if save.terminal_version is not None:\n'
    '            if not runner._terminal_worker.resident(save, job):\n'
    '                connector._invalid_completions.add(save.job_id)\n'
    '            continue\n')
(OUT/'gpu_completion_copy.py').write_text(gpu_copy)

runner = source('v1/worker/gpu/model_runner.py')
runner = once(runner, '        self.step_timing = StepTimingCollector()\n',
    '        self.step_timing = StepTimingCollector()\n        self._terminal_worker = None\n')
runner = once(runner, '            self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)\n',
    '            self.kv_connector = get_kv_connector(self.vllm_config, kv_caches_dict)\n'
    '            connector = getattr(self.kv_connector, "kv_connector", None)\n'
    '            if getattr(connector, "supports_versioned_completions", False):\n'
    '                from vllm.v1.kv_offload.gb10_terminal_worker import TerminalWorker\n'
    '                self._terminal_worker = TerminalWorker(connector, self)\n')
runner = once(runner, '            completion_hook = getattr(connector, "prepare_completion", None)\n',
    '            if getattr(connector, "supports_versioned_completions", False):\n'
    '                assert self._terminal_worker is not None\n'
    '                self._terminal_worker.before_requests(scheduler_output)\n'
    '            completion_hook = getattr(connector, "prepare_completion", None)\n')
runner = once(runner, '            self.block_tables.apply_staged_writes()\n',
    '            self.block_tables.apply_staged_writes()\n'
    '            if self._terminal_worker is not None:\n'
    '                self._terminal_worker.after_requests(scheduler_output)\n')
runner = once(runner, '        return async_output\n\n    def take_draft_token_ids',
    '        if self._terminal_worker is not None:\n'
    '            self._terminal_worker.capture(input_batch, sampler_output.sampled_token_ids, num_sampled)\n'
    '        return async_output\n\n    def take_draft_token_ids')
(OUT/'model_runner.py').write_text(runner)

launch = (ROOT/'launch_rank.py').read_text()
launch = once(launch, "paging = cfg.get('kv_paging', {})\n",
    "paging = cfg.get('kv_paging', {})\n"
    "async_parking = bool(cfg.get('async_scheduling', False))\n"
    "terminal_budget = int(paging.get('async_terminal_snapshot_bytes', 0)) if async_parking else 0\n"
    "kv_budget = int(paging.get('kv_bytes_per_rank', 4294967296))\n"
    "if async_parking and not 0 < terminal_budget < kv_budget:\n"
    "    raise ValueError('Async terminal reservation must fit inside the KV memory budget')\n"
    "kv_budget -= terminal_budget\n")
launch = once(launch, "str(paging.get('kv_bytes_per_rank', 4294967296))", "str(kv_budget)")
launch = once(launch, "  'native_completion_cache':paging.get('native_completion_cache',False),\n",
    "  'native_completion_cache':paging.get('native_completion_cache',False),\n"
    "  'async_terminal_snapshot_bytes':paging.get('async_terminal_snapshot_bytes',0),\n")
launch = once(launch,
    "args += ['--no-async-scheduling', '--scheduler-cls', 'vllm.v1.core.sched.gb10_parking_scheduler.GB10ParkingScheduler']",
    "args += ['--async-scheduling' if async_parking else '--no-async-scheduling', '--scheduler-cls',\n"
    "         'vllm.v1.core.sched.gb10_parking_scheduler.' +\n"
    "         ('GB10AsyncParkingScheduler' if async_parking else 'GB10ParkingScheduler')]")
(OUT/'launch_rank.py').write_text(launch)
print('Candidate source assembled; deployment unchanged.')
