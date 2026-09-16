# MTP3 with event-triggered completion checkpoints

User requested returning to MTP3 and removing expensive periodic decode checkpoints. The baseline-only MTP3 load was stopped at the user's correction; no new baseline measurement is being awaited. The candidate preserves existing scoped deterministic kernels, concurrent batching, 1920-token cache retention, 7680-token reservations, and full transfer verification.

Changes:
- Completion snapshots retain the actual computed accepted boundary rather than rounding down to 64.
- Connector metadata no longer requests intermediate decode snapshots.
- Worker captures recurrent state only for actual completion/parking save events, before slot reuse.
- Lookup accepts exact unaligned completion records.
- Restore schedules an initial partial arithmetic chunk ending at the next absolute 64-token boundary (or prompt end), then resumes normal canonical chunks. Insufficient batch budget defers the whole partial chunk, preventing concurrency from changing that split. This is a batched chunk, not a per-token or per-request GEMV loop.

CPU gates passed: 40 accepted-state/layout cases including refusal of unavailable historical state, 1260 partial-grid scheduling cases, and an ordinary-decode metadata no-op check. Both-node candidate source hashes verified. Live completion/stop/C4 gates passed; see RESULT.md and FINAL-AUDIT.json for measured performance and remaining disk/suspension coverage limits. No universal cold-versus-cached numerical equivalence is asserted.

The copied source remains in experiments/completion-triggered-20260914 and is mounted explicitly by both rank configurations. before-r0/r1.json preserve exact prior J/MTP5 configs; mtp3-baseline-r0/r1.json retain old checkpoint behavior at depth 3 for rollback if needed. Do not revert to MTP5 implicitly.
