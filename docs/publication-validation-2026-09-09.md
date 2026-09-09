# Publication and packaging validation — 2026-09-09

The public source import corresponds to the selected deployment after the
recurrent-cache fix. The existing service was not restarted or switched to
the new checkout during publication.

- All 21 maintained runtime modules match the validated deployment source
  bytes. Baseline/source SHA256 manifests are committed in the server project.
- A clean checkout with its pinned submodule exported every source file,
  verified the selected 98,304-token vocabulary and rendered the service units.
  Generated files, host configuration and verification records remain ignored.
- Intercepted launch commands on both ranks retained identical server arguments,
  environment variables, mount destinations and modes. The mounted file contents
  matched, including the entrypoint and selected draft IDs. Checkout paths and
  an isolated runtime-cache directory were the intended host-path differences.
- `systemd-analyze verify` accepted the generated head and worker units.
- Native CPU and mapped CUDA helpers built successfully against the pinned
  image. The resulting image passed cache metadata/lifetime regressions, packed
  row gathers, differential CPU hashing, row-cache stress and TP draft-ID tests.
- Synthetic NVFP4 and FP8 tables passed cross-shard byte comparisons, mmap
  attachment and scale-preservation checks using Mia's extended builder.
- An optional GPU transport test was attempted with the model resident. Its
  fallback worker failed a CUDA IPC allocation with an out-of-memory error.
  The isolated test container was removed; the production health check returned
  HTTP 200. This attempt does not establish GPU transport correctness for the
  rebuilt image. Repeat GPU/full-model validation with the service stopped.

The runtime implementation has the earlier full-model capacity/functional
evidence in the cache-fix report. The newly packaged image has build and
CPU-regression coverage; it has not been adopted by the running service.

Publication review covered both files and Git history. Private workload outputs,
host credentials, model weights and captures were excluded from the public
imports. Gitleaks' only remaining matches were a Python identifier assignment
and a verified public tokenizer SHA256; exact finding fingerprints are listed
in the respective `.gitleaksignore` files with explanations.

The complete original local experiment history is retained separately. The
public import uses a GitHub noreply author address and records source commit
IDs in its provenance files.
