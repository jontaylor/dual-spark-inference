# Current deployed state — V2 aligned completion cache repair

V2 is live on both ranks, healthy, MTP3 enabled. Both deployment configs and all 15 runtime overrides/rank hash-verified. Scheduler is unchanged from validated MTP scheduler. V1 speculative clipping was rejected after mixed-C8 failures; do not redeploy it.

Validation PASSED: original full C1/C4/seed20 requests; mixed16 responses with peak8 live; warm C1/C4; uncached C1/C4; independent exact-finish vs retained-shadow checkpoint restored tokens and scores. CPU snapshot checks cover crossing1/2/3, immutable recurrent/ring state, cleanup and invalid-state rejection.

Cache-history-independent determinism remains unresolved: cold/warm and restored/uninterrupted differ. Cross-restart equality not established. Do not claim unconditional determinism.

Controlled benchmark: same7393-token prompt/16output/C4; partial cached4800 gives15.94–16.02 aggregate tokens/s; repaired cached7360 gives33.35–34.86. About2.13x throughput,98.73% less recomputed prompt work. Independent salts used for each partial phase.

All seven campaign PIDs identity-checked and SIGCONT resumed at02:38:29UTC, children before supervisor. Owner task notified; sampling/retries preserved. Current owner01a09b9f-c6c7-7bb0-8c25-b22748f43388 on192.168.0.167. campaign-resume-v2.json contains results.

observe_live.py records post-resume interval metrics. Initial long cold prompts are rebuilding cache; do not interpret lifetime counters from validation/startup as warmed production reuse. See REPORT.md and post-resume-metrics.jsonl for evidence. Keep successful repair running.

Final observation: 21 real follow-up completions reused99.37%; post-initial-prefill interval190s averaged101.7aggregate generation tokens/s. Full resumed interval incl cold rebuild83.21%reuse/63.4tokens/s. Server healthy, workload resumed, repair left running.
