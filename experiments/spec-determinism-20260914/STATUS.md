# Deterministic MTP investigation

Candidate with three draft tokens is starting; whole-model correctness and speed validation remain pending. Previous stable configuration is experiments/targeted-determinism-20260913/candidate-final-r0.json and r1.json.

Short tests passed on the serving worker GB10 in separate processes before restart:
- Stock fused speculative recurrence: C1/C4 matches for FP32/BF16 state; four-token versus four single steps fails for BF16 state.
- Candidate uses packed-decode normalization/reduction arithmetic, one warp, and rounds intermediate recurrent state to cache dtype after each token.
- Candidate matches C1/C4, repeated single steps, and existing packed decode bit-for-bit in outputs and states for both cache dtypes.
- Real convolution plus recurrence matches packed decode and C1/C4. Rejection/resume at each acceptance length1-4 matches ordinary decoding in output, recurrent state and convolution history.
- Full GDN core speculative decode matches alone, C4, mixed with3 prefills and C4 mixed with2 prefills (330-token prefills; real kernels; synthetic inputs; both state dtypes).

Scripts and JSON evidence are here. No per-request GEMV or sequence loop was added to serving. The temporal loop is inside the batched recurrent GPU kernel. Scoped existing GEMMs and fixed GDN norm/prefill behavior retained.

Head standalone test initially ran, then a second process hit CUDA allocation failure while the representative background load had11 active requests and86% KV use. API remained healthy. Candidate short tests instead ran successfully on rank1.

Both launcher dry-runs passed, verifying runtime hashes. Restart command exceeded its60s client timeout while the old server drained, but systemd continued and started the new service successfully. Background-load task has been asked to pause new arrivals for isolated validation and retain settings/retries. A queued message is not an acknowledgement.

Final serving validation passes86 requests including measured C8, complete responses and seed/cache-salt repeats. All9596 selected tokens are reported maxima. Live configs/hashes verified. MTP left enabled; supervisor and all8 child campaign processes resumed with PID-start identity checks. See REPORT.md for measured speed and cross-spec-off equivalence limitation.
