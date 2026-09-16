# Targeted batch-invariance candidate

Full-server validation pending. Candidate deployed on both ranks; no global batch-invariant mode and no request serialization. Speculation remains disabled.

## Arithmetic changes

HC BF16 projections and GDN in_proj_qkvz/in_proj_ba/out_proj use vLLM persistent Triton GEMM with fixed M/N/K tiles 128/128/64, 8 warps, 3 stages. No M-dependent tuning. GDN gated RMS uses ROWS_PER_BLOCK=1; fused decode norm is disabled to retain that arithmetic route.

`projection-replay.json` proves HC C1 versus C4 equality using one M=4 launch per projection (K/N 10240/336 and 320/10240). Captured M330/M991 HC replay also matches. GDN projection and gated-norm primitive tests remove observed differences. Gated norm inputs are constructed from captured output-projection inputs; these are not captured pre-norm tensors.

## Separate execution changes

Adapted PR #49827 (debb4ba43485a3efa41e6c81180e004b81be6a16): same packed recurrent path for pure/mixed cached decode; FlashInfer use_cp=False; nonfinal prefills end on absolute 64-token boundaries. Existing 1600-token cache stops retained. Completion lookup skips snapshots whose boundary is not divisible by64, reducing completion-cache hit coverage. Normal block-prefix caching remains enabled.

Baseline routing control changes output for FP32 and BF16 states; candidate routing matches output/state. Convolution and fresh prefill are stubbed in this isolated route test. Real Triton prefill 192 versus 64+64+64 matches; negative control73+55+64 differs. C1/C4 prefill matches. FlashInfer check is argument-level only (GB10 uses Triton).

Scheduler checks cover budget73, block1600 materialization and rejection of unaligned starting states. Completion checks cover aligned restoration, unaligned rejection and unaligned pending records not blocking.

PR #45819 (c6cb0851075013a0e14cae08bad688c2bb83f055) also inspected; its per-sequence GEMM/recurrence loops were deliberately not ported.

Both upstream PRs were open when inspected. Their published validation does not cover this TP2/EP/quantized/prefix-cached deployment; full serving checks are required here.

## Reproduction

Scripts and JSON evidence in this directory. Lab container: gb10-determinism-lab. Run GPU checks with docker exec python3 /e/projection_replay.py and /e/gdn_execution_checks.py [baseline]. Scheduler and completion checks also run in the lab.

Per-rank original configs: config-before-r0.json and config-before-r1.json. Candidate configs: candidate-r0.json and candidate-r1.json. Runtime files are mounted with SHA256 verification. Existing FLA override retained. Diagnostic hooks dormant until probe control enables them.

## First full-server result and next localization

Initial HC/GDN candidate served successfully. Three serial responses matched tokens, all returned scores and 702-token full responses. Every concurrent response diverged by token index2 (zero-based), with scores differing at index0 or1. Long probe was deliberately interrupted once failure was established; no completed serial-after or fixed-seed result is claimed for this candidate.

Four-token `captured-next` trace: matching prompt and prefill inputs. All captured full GDN0/2 projection/core-output inputs/outputs and initial states equal. First last-prefill-token module difference is layer3 self_attn output (maximum0.001953125); its input is equal. See traces-r0/layer-analysis.json and full-analysis.json.

QSA follow-up primitive checks:
- QKV BF16 with captured final input repeated across M: M1/M4 has9 changed values, maximum0.0078125; fixed GEMM equal.
- Index QK BF16:208 changed values, maximum0.015625; fixed GEMM equal.
- Output projection with synthetic input and checkpoint weights:3 changed values, maximum0.00006103515625; fixed GEMM equal. This is not a captured o_proj-input replay.
- QSA sparse attention explicitly varies BLOCK_N, NUM_SPLITS and warps with batched row count. Identical synthetic q/cache/indices produce different C1/C4/M330/M991 results; fixed BLOCK_N64, target splits8, warps2 matches across all four. FP32 reference maximum absolute error0.00005982816219329834. See qsa-attention-check.json.
- Existing QSA index scoring and top-k selection pass C1/C4/M330/M991; unchanged. See qsa-index-check.json.

Second candidate adds only QSA's three BF16 projections and fixed sparse-attention reduction; deeper optional QSA trace captures q/k/v/gate and attention output. Configs candidate-qsa-r0/r1.json. Full-server retest pending.

The original QSA reduction source hash is faa8d358c79745f304edd363e4da21992e4cf015a22316b14980500bd199a0ad, identical to upstream model-support commit e126687a9a (#53896). Its M-dependent split schedule was not introduced by our local top-k backport.

## QSA candidate live trace

Four-token `captured-qsa` test completed without request errors. First concurrent request diverged at token index2 while sharing a batch with fresh prefills; three fresh-prefill requests matched the four token IDs but not scores. First concurrent prefill hit1600 cached tokens; the serial reference had zero cached tokens and also split at1600.

Layer3 QSA fix removes its observed output difference. The last-prefill-token trace first differs in layer45 linear_attn output, maximum0.0078125. This is a last-row localization, not proof that every earlier row is equal. Uninitialized indexer output-buffer contents at input hooks are not treated as arithmetic failures. See traces-qsa-r0/step-analysis.json.

Added opt-in deep diagnostic overrides (candidate-deep-r0/r1.json): force eager only while trace control enables it, hash all rows of every module's row-shaped inputs/outputs, choose full-tensor layers from control, and capture GDN core outputs. Normal serving still dispatches CUDA graphs. This diagnostic reload changes no numerical kernel.

Real convolution + packed recurrence passes for FP32 and BF16 states at actual per-rank heads H8/HV24. Full GDN core test (real convolution/gating/prefill recurrence, synthetic inputs and valid nonzero cache slots) passes C1 versus C4 prefills and one-decode+three-prefill mixed batches at330 tokens per prefill. Output and state equality checked; see gdn-real-conv-check.json and gdn-full-core-check.json.

## All-row/eager localization and fourth candidate

`captured-deep` and `captured-dense` traces show matching prefill module-output hashes but decode differences with equal inputs at layer0 MoE router, shared-expert gate/up and down projections, plus PLE key/value projections. Actual BF16 runtime weights and captured input replay reproduces the projection differences; fixed GEMM removes them. Scalar shared-expert gate passed and remains unchanged. See dense-projection-check.json.

Some concurrent prefills have equal final hidden states but differing logits (maximum0.03125). Checkpoint-head primitive changes792 values across four rows C1/C4. Fixed128 and fixed16 head kernels both match C1/C4, and match each other. Direct stock-head replay differs from captured rank0 logits in23 values by at most0.0078125, so it is not claimed as an exact serving-output replay. See head-projection-check.json.

Head kernel fixes M/N/K tiles to16/128/64, 4 warps, 3 stages. C1/C4/C8 equality checked. Isolated benchmark: fixed16 2.82/2.84/2.87ms at C1/C4/C8, stock3.72/2.70/2.81ms. Not an end-to-end throughput result.

PLE norm_key/norm_query/norm_conv and gate sum pass60 captured-row replication checks; unchanged.

Fourth candidate configs: candidate-dense-r0/r1.json. Adds scoped BF16 methods to MoE router/shared-expert projections, PLE key/value and final LM head. Retains embedding-method inheritance for loader/tied-weight semantics. Normal graph capture and request capacity retained. Optional local trace control can test extra named BF16 linears in eager mode and restores them when disabled; no global GEMM override or external administrative API added. Full-server retest pending.

Head replay clarification: Qwen4Exp's low-latency dispatcher handles ParallelLMHead via Qwen4ExpLowLatencyEmbeddingMethod as well as ordinary linears. The `stock` head primitive in the laboratory used torch.nn.functional.linear (cuBLAS), not that serving-specific dispatcher; this explains why it is not an exact serving reference. FixedBF16EmbeddingMethod is a distinct subclass and bypasses the dispatcher's exact-type gate.

## Remaining scalar gate found; live prospective fix

Normal-graph fourth candidate fixes the mixed decoding request (hidden states, logits, scores and four tokens equal) but remaining prefills still differ. All-row `captured-prefill-full` trace finds the first changed output at layer44 shared_expert_gate, not GDN45. The last-row GDN45 difference is propagation from an earlier prefill token.

`captured-scalar` + actual runtime-weight replay pin this down: identical330-token input slice vs its position inside M991; stock scalar projection changes row122 by0.015625, fixed GEMM changes zero values. See scalar-projection-check.json. The earlier layer0 scalar probe was insufficient to exclude this operation class.

Live opt-in eager test applies FixedBF16LinearMethod to the48 named shared_expert_gate modules, with a separate cache salt. `scalar-pilot` completes serial3/concurrent4/serial3: all10 match token IDs and complete returned score records, with no errors. This is a prospective fix test, not yet the permanent normal-graph serving configuration. Longer scalar-128 experiment running.

Prospective scalar-128 and scalar-128-seed both pass: all20 complete117-token responses have exact token-ID and returned-score equality, including across experiments. Permanent scalar fix is model.final.py; final configs candidate-final-r0/r1.json.

First final startup was rejected before model loading by the disk reserve check (48GiB cache +8GiB free margin). Deduplicated47 identical .pt paths by verified SHA256 hard links, preserving every path/content and reclaiming4.666GiB; removed stopped serving/lab containers after archiving their trace data. Free disk57GiB; cache budget unchanged. See evidence-deduplication.json. Lab container has been removed; its scripts and results are preserved. Final startup retried; run_final_ready.py is checking normal full responses and seed repeat.

## Final deployment validated and left running

Both final configs and all 13 mounted source hashes per rank match the candidate. Normal-graph original-prompt tests pass 40 responses across ordinary/fixed-seed and unique-cache-salt ordinary/fixed-seed runs: every 125-token output, prompt ID sequence and returned score record matches. All 20 unique-salt responses report zero cached prompt tokens.

Mixed test `mixed-20260914T003356Z` passes 16 requests at four prompt lengths (36/146/1826/3526), with 96 generated tokens each. All concurrent/after responses match their prompt-specific serial tokens and scores. Metrics confirm eight simultaneously running requests. Final API health 200; service left running without another restart. See REPORT.md and final-*-validation.json for results and scope limits.
