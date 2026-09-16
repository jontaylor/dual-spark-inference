# Experiment source archive

The maintained deployment selection is **original external PLE workers**,
s32/b8192/t1024, TP2/MTP3, with unrestricted CPU affinity. See
[the restoration record](ple-external-restore-20260916/README.md) and
[the deployment handover](../docs/kv-paging-handover.md).

This directory preserves investigation code, candidate implementations, tests,
patches and reports from the September 12–16 campaigns. An experiment's presence
does not mean that it is deployed, recommended, or independently validated.
Many scripts assume the original two-host environment and deliberately perform
deployments or live load changes: read them before running them. Historical
paths and process IDs are records, not discovery mechanisms for current state.

The [complete PLE backend comparison](ple-backend-comparison-20260916/FINAL_ANALYSIS.md)
found no demonstrated throughput winner on its fixed warm 12-request workload.
Its [critical-path review](ple-critical-path-20260916/ANALYSIS.md) explains the
remaining attribution questions and implementation choices. The
[scheduler analysis](empirical-work-score/smooth-final/FINAL_ANALYSIS.md) documents
the selected scheduler settings and their quality limitations.

Large tensors, model data, raw request/response captures, logs, compiled binaries
and host-local configuration backups remain outside Git. Reviewed compact
evidence and the current example configuration are published explicitly.
Raw captures are retained locally; some historical report links therefore refer
to unpublished local files. Sources alone do not reproduce every historical
run without those inputs and the documented model/image/hardware dependencies.

Files mounted by a running deployment are frozen: make a new candidate copy
instead of editing them in place. The in-process and comparison implementations
remain archived here; they are not part of the restored external-only runtime.
