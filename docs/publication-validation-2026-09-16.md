# September 16 deployment and experiment publication

The production selection is the restored original external-only PLE path.
[Restoration evidence](../experiments/ple-external-restore-20260916/README.md)
records the successful two-host configuration/source verification and API smoke.
The warm repeated comparison established no demonstrated throughput winner;
maintainability motivated the user's selection.

This publication includes the pending deployment launcher/paging integration,
the current portable example configuration, runtime overlay sources, experiment
code and reports, and compact PLE comparison/restoration evidence. Captured
tensors, raw workload outputs, compiled libraries, credentials and local config
backups remain outside Git. The experiment archive is historical research code,
not a statement that every candidate is production-ready.

The server dependency is published at `jontaylor/vllm-gb10`, branch
`gb10-pressure-paging`, revision `318e06e84d2aeb70a240531705ae67fa3f977b7a`.
The deployment submodule now pins that same revision. Every source exported by
prepare_paging.py was independently read from that commit and checked against
paging-manifest.json. Every active runtime override staged in this repository
matches the SHA256 recorded in the deployed configuration. Existing image and
model provisioning remains subject to the earlier build documentation.

Remaining uncommitted in-process changes in the separate server checkout are
preserved as an [apply-checked source patch](../experiments/ple-in-process-source-20260916/README.md).
That checkout was not modified; the production server does not enable the patch.

Validation performed for this publication:

- Both external-only launch dry runs passed, followed by live source/mount,
  process, parameter and unrestricted-affinity checks on both nodes.
- Health, authentication, both model aliases, four concurrent arithmetic
  responses, tool calling and streaming passed after restoration.
- The four existing paging/deployment utility tests passed in an isolated
  pytest environment. Published Python archive sources passed syntax parsing.
- Gitleaks scanned the staged deployment/archive content and both previously
  unpublished server commits. Exact reviewed false positives are documented in
  .gitleaksignore: Python attribute assignments and prose token/timing values.
- Archived source/diff snapshots retain original bytes, including historical
  whitespace, so published hashes and patches remain meaningful. They were not
  reformatted or revalidated as active deployment candidates.

The dependency's earlier 46-test pressure-only and 56-test-per-node memory-cache
validation are recorded in pressure-paging-rollout.md and
memory-completion-rollout.md; those historical suites were not rerun merely to
publish their already tested commits. No standalone GPU test was launched
alongside the serving model. Publication does not broaden the documented
performance or model-quality claims.
