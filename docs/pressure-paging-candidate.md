# Pressure-only disk paging candidate

Prepared on 2026-09-13. Not deployed; two-rank GPU validation is still required.

Source: `/home/jon/vllm-gb10-pressure-paging`, branch `gb10-pressure-paging`.
Committed source: `41a8cd599a710eae470d725e7499de65ea42e843`.
Deployment staging: `/home/jon/dual-spark-inference-pressure-paging`.
The outgoing server's files and configuration have not been changed.
The candidate bundle has been generated and all 11 manifest hashes verified.

## Behavior

Set the following fields inside the existing `kv_paging` configuration, preserving
all other deployment settings:

```json
{
  "disk_write_policy": "pressure",
  "save_timeout_seconds": 300
}
```

The launcher keeps `eager` as its compatibility default. Pressure mode disables
automatic block/boundary stores and completed-request disk snapshots, even if
`completion_checkpoints` is true. The completion capture machinery remains
available for explicitly requested pressure snapshots. Ordinary resident prefix
caching continues; exact completed-conversation disk reuse is no longer provided.

A failed growth reservation quiesces a request. Only one pressure snapshot is
created at a time. The scheduler preserves the GPU blocks and worker slot while
the full current hybrid state is saved. Both ranks must acknowledge before the
snapshot is pinned and the request's resident reservation is released. Restoration
uses that request-bound descriptor, including recurrent state and compression
rings, rather than requiring a previously persisted aligned prefix.

Cancellation waits for transfer completion. Invalid captured state fails closed.
The save deadline is measured from that request's actual save start; time waiting
for another request's serialized save does not consume its deadline. This is a
fixed configurable timeout, not byte-progress-based timeout tracking.

## Validation and remaining gate

The isolated CPU suite runs in the deployment image without GPU devices or network:

```bash
cd /home/jon/vllm-gb10-pressure-paging
/home/jon/vllm-gb10-kv-paging/.venv/bin/python tools/pressure_paging/test_cpu.py
```

46 tests passed, covering existing eager behavior plus pressure store suppression,
aligned/unaligned snapshot metadata, two separate rank acknowledgements, pinning,
restore selection, retained GPU ownership, cancellation and invalid-state handling.
These tests do not establish CUDA state correctness or decode performance.
All source pre-commit checks passed, including mypy, Ruff and commit sign-off.
The modified deployment scripts pass Python compilation and whitespace checks.

After the current requests naturally finish, the candidate still needs an isolated
two-rank GPU canary: force reservation pressure, verify physical save/release/restore,
compare resumed generation with uninterrupted controls, and verify zero new store
bytes during low-pressure prefill/decode/completion. Include speculative rejection,
alignment transitions, cancellation and transport failure. Do not deploy as the
normal service if these checks fail.

## Packaging and cutover

The launcher now mounts the inherited offloading scheduler as an explicit overlay,
and the bundle manifest includes it. Source changes must be committed before
running `prepare_paging.py --server-root /home/jon/vllm-gb10-pressure-paging`.
The staging launcher includes a copy of the outgoing uncommitted launcher changes;
they have not been discarded. Its referenced runtime configuration, virtualenv,
model artifacts and experimental override files must be resolved from the actual
outgoing deployment before starting it. This staging directory is not a newly
configured service.

Do not pause admissions or stop the running workload. Wait for running, waiting,
parked and transferring requests to drain naturally. Recheck immediately before
the coordinated head/worker stop; if new accepted work appears, wait for it too.
Preserve the outgoing manifest, overlays, launcher and configuration for rollback.
Start matching candidate bundles on both ranks only during the drained canary.
Revert to the outgoing bundle if correctness or low-pressure store checks fail.
