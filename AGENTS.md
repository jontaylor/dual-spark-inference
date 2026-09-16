# Representative load coordination

The user has authorized coordination with their existing Codex task on the
`artifact-agent-orchestration` server for representative inference loads.

- SSH host: `jon@192.168.0.167`
- Remote workspace: `/home/jon/artifact-agent-orchestration`
- Current Codex task title: **Review latest planner experiments (3)**
- Current task ID: `01a09b9f-c6c7-7bb0-8c25-b22748f43388`
- Contact method: run the remote Codex CLI's `queue` command over SSH. This queues
  a message for the existing task; it does not wait for a reply or confirm that
  requested changes have been applied.

Example:

```bash
ssh jon@192.168.0.167 'codex queue --thread 01a09b9f-c6c7-7bb0-8c25-b22748f43388 --message "Please report the current representative-load status, active sampling settings, and load/control paths. Keep the load unchanged while reporting."'
```

For dynamically generated messages, use `subprocess.run` with an argument list
and `shlex.join` for the remote command; do not treat JSON encoding as shell
quoting:

```python
import shlex
import subprocess

message = "Please report the current representative-load status."
remote_command = shlex.join([
    "codex", "queue",
    "--thread", "01a09b9f-c6c7-7bb0-8c25-b22748f43388",
    "--message", message,
])
subprocess.run(
    ["ssh", "jon@192.168.0.167", remote_command],
    check=True,
    timeout=30,
)
```

This task can help run representative planner/code workloads and change client
sampling settings. Coordinate the workload, concurrency, thinking mode,
temperature, top_p/top_k, duration, and evidence/output paths before comparing
inference configurations. Confirm actual settings or results rather than
assuming a queued request has taken effect.

As of September 16, the fixed PLE comparison completed ten cycles and its client
is intentionally held idle. Do not resume the older automatic load based on the
historical instructions below. Current records are under
`/home/jon/artifact-agent-orchestration/.artifact-agent/ple-comparison/20260916-fixed-decode`;
coordination summary: `/tmp/ple-repeating-load-status.json`. The original
external-only server implementation was restored after the comparison.

As of 2026-09-12, the load automatically retried through inference-server
restarts. The user authorized those restarts and asked that the load remain
running. Preserve its sampling, cadence, and retries unless the active experiment
calls for changing them; notify the task about planned restarts and restoration.

Historical load directory (verify it is still current before using it):

```text
/home/jon/artifact-agent-orchestration/.artifact-agent/baseline-comparison/20260912T122137Z-staggered-baseline-load
```

Its `sampling-control.json` used variant `baseline` and epoch
`post-draftcal-random-baseline` during the September 12 cost profile. The active
thinking workload then used temperature 1, top_p 0.95, and top_k 20. These are
historical observations, not permanent defaults. If the task ID becomes stale,
locate the existing task by its title/workspace before contacting a replacement.
