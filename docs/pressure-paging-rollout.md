# Pressure-only paging deployment — 2026-09-13

Applied to both TP ranks following the user's instruction to restart immediately,
superseding the earlier natural-drain requirement. In-flight work was interrupted.

Both running containers use `disk_write_policy=pressure` and the explicit
offloading scheduler overlay from source revision
`41a8cd599a710eae470d725e7499de65ea42e843`. All 11 candidate manifest hashes were
verified before startup. Existing model, PLE, memory budgets and runtime overrides
were preserved. `save_timeout_seconds` is 300.

The coordinated head/worker restart completed. Health returned HTTP 200. A real
completion with 3,310 input tokens and 64 generated tokens finished successfully;
connector store and load counters remained zero throughout the request, including
completion. This crosses two 1,600-token boundaries that previously triggered
automatic saves. Normal clients subsequently resumed submitting requests.

Evidence and per-node rollback archives:
`/home/jon/pressure-paging-rollout-20260913/` on each node. The head directory holds
`serving-check.json`, the candidate bundle, installation and verification scripts.
The archives are `rank0-before.tar.gz` and `rank1-before.tar.gz` respectively and
include outgoing configuration, launcher, manifest and overlays.

Prior validation: 46 CPU lifecycle tests and all source pre-commit checks passed.
This deployment check establishes normal GPU generation and low-pressure store
suppression. Forced-pressure GPU save/restore correctness and a controlled decode
performance comparison remain outstanding; no numerical or throughput improvement
claim is made from this serving check.

To roll back, stop the coordinated head service, restore the corresponding
`rankN-before.tar.gz` into `/home/jon/dual-spark-inference-kv-paging` on each node,
then start the head service. The outgoing launcher does not mount the additional
offloading scheduler file, so a leftover unreferenced copy is inactive.
