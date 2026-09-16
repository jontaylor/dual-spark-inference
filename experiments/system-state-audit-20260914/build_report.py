import json,datetime
from pathlib import Path
p=Path(__file__).parent;root=Path.cwd();h=root/'experiments/performance-until-10am-20260914'
def link(f):return f'[{Path(f).name}]({root/f})'
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(x).replace('|','\\|').replace('\n',' ') for x in r)+' |\n' for r in rows)+'\n'
def fmt(v):return json.dumps(v,ensure_ascii=False,separators=(',',':'))
r=['# Auditable system-state delta — 14 September 2026\n', '''## Scope and evidence boundary

The six-hour A→J experiment is recorded in `experiments/performance-until-10am-20260914`: the first monitoring sample is 02:54:45 UTC, and observation ended09:00 UTC. The exact user-session start timestamp is UNKNOWN. **A is the saved pre-experiment configuration, not a reconstructed pristine/default vLLM baseline.** A already contained15 overrides. J had18. The currently serving later GPU-completion candidate has22 manifest overrides. It is incorrect to describe J as the server running now.

This audit includes A→J and subsequent documented changes through the live capture. Historical exact process environments/argv at A are UNKNOWN unless preserved in the historical runtime artifacts; configuration snapshots establish intended configuration, not a retroactive process dump. Both current ranks were read directly using Docker inspect and /proc. No service configuration or workload was changed for this audit.

The live capture records containers started2026-09-14T14:19:03.531637462Z (rank0) and2026-09-14T14:19:03.583698975Z (rank1). Both were running. All22 manifest override hashes match inside each container. This is **source identity verification**, not a claim that inference correctness is fully verified.

Raw live captures: [rank0](live-r0.json), [rank1](live-r1.json). Readable exact argv and environments: [rank0](live-r0.txt), [rank1](live-r1.txt). API keys/credential values are explicitly redacted. No other argument/environment value is intentionally omitted from these attachments. The collection helper's own transient process appears in the raw /proc inventory; it is not an inference worker.

## 1. Exact saved configuration: before → J → now

All flattened non-source configuration fields are listed below, including unchanged settings. Source overrides are enumerated separately. NOT SET means absent from that configuration, not necessarily disabled. Actual current argv in section6 takes precedence over interpreting an absent field.
''']
why={
'block_size':('1600→1920 for MTP5 state/ring layout; retained after MTP3 return.','MTP5/1600 resolved3200 and failed12-slot ring compatibility; MTP3/1600 compatible. Current padded state page20.30% over raw largest state versus0.25% at1600. Whole-system isolated effect UNKNOWN.'),
'cudagraph_capture_sizes':('Added160/192 for32 sequences ×6 MTP5 target rows; retained now.','Isolated memory/speed consequence UNKNOWN.'),
'mtp_tokens':('3→5 to test more accepted draft tokens; returned3 at user direction.','H/I3/J comparison below; not an isolated MTP speedup. Cross-configuration equality not established.'),
'prefix_cache_retention_interval':('1600→1920 with layout; subsequently0 for semantic-only native retention.','Native-only coarse replay regression measured; exact GPU completion path later restored. Isolated retention effect UNKNOWN.'),
'kv_paging.reservation_tokens':('Absent A; baseline source default32768. J introduced7680 to reduce advance reservation.','Growth across7680/15360 verified; J vs I3 writes lower, throughput also lower; confounded comparison.'),
'kv_paging.native_completion_cache':('Later addition: GPU completion ownership through ordinary free/LRU pool.','75/80 sampled follow-ups exact endpoint; mean estimated replay185.75;120.35s167.08tok/s, zero disk; different workload from H/I3/J.')}
rows=[]
for x in json.loads((p/'config-delta.json').read_text()):
 w,c=why.get(x['setting'],('Unchanged saved value.','No isolated effect measured.'))
 rows.append([x['setting'],fmt(x['before']),fmt(x['J']),fmt(x['now']),w,c])
r.append(table(['Setting','Before A','J at09:00UTC','Now','Why changed','Measured consequence'],rows))
r.append('''Configuration sources: [A](config-A.json), [J](config-J.json), [now](config-now.json); machine-readable [delta](config-delta.json). Rank-specific historical originals remain `baseline-r0.json`, `rollback-r0.json`, `candidate-H-r0/r1.json`, `candidate-J-reservation-r0/r1.json` in the historical experiment directory. Exact rank1 pre-A environment and launch identity: UNKNOWN. Historical worker configs omit the head-only request-template/API options; current actual rank differences are in section6.

Additional behaviour not represented by a scalar config delta:

''')
r.append(table(['Behaviour','A','J','Now','Consequence'],[
['Completion capture','64-token decode shadows, aligned completion endpoint','Same periodic64-token mechanism','Completion/suspend event capture; no periodic decode capture','Earlier head-thread sample10.3% checkpoint capture vs event1.55%; different loads, not causal speedup.'],
['GPU cache ownership','Separate resident completion allocations/copy path','Same resident model with inherited full-page keys','Borrow attention/ring pages; allocate normalized recurrent pages; zero-ref cached LRU after ACK','Below-pressure follow-up disk bytes0; small GPU-state copy still exists.'],
['Disk storage','Fresh per-slot payload','Exact-content dedup + inherited complete-page sharing','Dedup retained; actual allocator pressure triggers backing','J requested178.581GB vs82.882GB payload over its observation; no claim all unused writes predictable.'],
['Transfer integrity','CPU checksum and GPU readback','Both enabled finally; readback off transiently in paired H test','Both retained for disk','Readback off did not demonstrate aggregate benefit.'],
['Scoped GEMM plans','128×128×64 general projection plan','HCdown32×32, BA16×16 for M≤128; K64 fixed','Same as J','Synthetic geomean1.795× HCdown/2.678× BA; not end-to-end attribution.'],
['Scheduler','Custom synchronous parking scheduler','Same + configurable reservation','Same + event-capture scheduling and native eviction integration','--no-async-scheduling still active; isolated async trade-off UNKNOWN.']]))
r.append('''## 2. Every active override

The requested18 correspond to J and are rows1–18 below. Rows19–22 were introduced after J. “Baseline” below means saved A where available, or explicitly described pre-patch/native operation. It does NOT claim every complete file equals upstream apart from one described modification. Full function inventories and exact A→now unified diffs are attached in [override-inventory.json](override-inventory.json) and `override-NN-A-now.diff`; newly added override files are shown in full when A had no override. A full upstream-fork diff of the entire image is not available in this audit: UNKNOWN. The image already includes other patches beyond mounted overrides.

''')
info=[
('Backend/check_shared_mem','Image threshold102400 bytes excludes GB10 limit101376; A already patched.','Threshold101376; changes eligible FLA shared-memory plans.','GB10 kernel eligibility; isolated timing within~2%, output max abs0.00024414 in old test.','Pre-existed A; unchanged.'),
('GatedResidual.mix/combine/combine_and_mix','Original HC BF16 projections use shape-dependent matmul; A fixed.','Scoped fixed_bf16_matmul for HC projections.','Remove observed M-dependent HC score propagation.','Pre-existed A; unchanged.'),
('fixed_bf16_matmul; FixedBF16LinearMethod; FixedBF16EmbeddingMethod','A uses128/128/64 projection tiles; fixed16/128/64 head.','HCdown32/32 and BA16/16 at M≤128; others baseline; K64 constant.','Reduce fixed-GEMM overhead while retaining tested bit equality.','Existed A; changed in H; retained.'),
('QwenGatedDeltaNetAttention._forward_core and speculative/decode paths; ChunkGatedDeltaRule','Original routes choose batch-dependent implementations; A already uses fixed projections/norm, canonical routing, custom speculative recurrence.','Same source as A: fixed BF16 qkvz/ba/output, explicit fixed norm, deterministic recurrence route/use_cp=False.','Remove observed projection and recurrent execution-shape differences.','Pre-existed A; unchanged.'),
('calc_rows_per_block; layer_norm_fwd; RMSNormGated','Original row tiling depends on row count; A fixed.','ROWS_PER_BLOCK fixed1 in scoped gated RMSNorm.','Fixed reduction order across M.','Pre-existed A; unchanged.'),
('Scheduler.__init__; _mamba_block_aligned_split; _cache_aligned_split; scheduling/capture hooks','A fixed64 arithmetic grid, periodic shadow plumbing, MTP≤3 guard.','J guard≤5; now completion-event plumbing and partial initial restore chunk before canonical64 grid.','Allow tested MTP5 historically; remove periodic capture cost and support exact event endpoints.','Existed A; changed I3 and after J.'),
('CompletionCache.finish/lookup/allocate/update; capture_completion_states','A aligned64 shadows, separate completion snapshots; J inherited attention keys.','Exact event GPU capture, borrow history pages, normalize state, ACK publication, lookup pins across allocation.','Reduce redundant replay without periodic history copies; fix v1 lifetime race.','Existed A; changed H and after J; v1 superseded by v2.'),
('GPUModelRunner.execute_model/sample and trace hooks','Image runner lacks this diagnostic instrumentation; A already includes it.','Opt-in trace hooks retained. Normal mode uses usual graph dispatch.','Locate first differing tensors/score rows.','Pre-existed A; unchanged.'),
('begin/snapshot/scores/finish/install_layer_hooks/force_eager/apply_extra_linears','Custom diagnostic helper absent upstream; A includes it.','Opt-in tensor/row/hash capture and diagnostic wrappers.','Debug concurrency attribution.','Pre-existed A; unchanged. Trace control file absent on both ranks at audit; opt-in tracing disabled.'),
('Qwen4ExpQSAAttention / Qwen4ExpQSAFlashAttentionImpl','Original QSA BF16 projections and shape-dependent attention plan; A fixed.','Fixed QSA QKV/output/index projection methods and attention dispatch.','Remove earliest QSA tensor differences.','Pre-existed A; unchanged.'),
('qsa_sparse_paged_attention; split/merge kernels','Original sparse attention reduction schedule varies with shape; A fixed.','BLOCK_N64, target splits8, warps2.','Fixed sparse-attention reduction schedule.','Pre-existed A; unchanged.'),
('Qwen4ExpSparseMoeBlock; Qwen4ExpModel; Qwen4ExpForCausalLM.compute_logits','Original router/shared-expert/scalar gate/head use original BF16 matmuls; A scoped fixed.','Fixed router/shared-expert/scalar gate and head; quantized routed-expert paths retained.','Resolve remaining earliest differences including layer44 scalar gate and equal-hidden-state logits.','Pre-existed A; unchanged.'),
('Qwen4ExpPLELayer projection initialization/forward','Original PLE key/value BF16 projection matmuls; A fixed.','Scoped fixed projection method; existing PLE offload retained.','Remove observed PLE projection differences.','Pre-existed A; unchanged.'),
('fused_sigmoid_gating_delta_rule_update_kernel/update','Original speculative multi-token state arithmetic differs from packed ordinary decode; A patched.','Fixed warp/reduction/normalization/exponential arithmetic and per-token cache-dtype state rounding.','Match speculative recurrence primitive to repeated ordinary decode.','Pre-existed A; unchanged.'),
('GB10AlignedOffloadingConnector completion/provenance/metadata/pressure hooks','A custom aligned connector with separate snapshots; no pristine upstream equivalent.','J inherited-key cleanup; now GPU pool binding, native eviction metadata/fences, completion capture/recall integration.','Connect pressure offload and completion lifecycle safely.','Existed A; changed H and after J.'),
('DiskSlotManager; RankLocalDiskWorker._transfer; RankLocalDiskOffloadingSpec','A image implementation separate resident snapshots and fresh physical writes.','Dedup store, GPU capture/recall, normal-pool source ownership/pins, spill ACK leaves new block owner intact.','Avoid duplicate writes/copies and protect ownership under reuse.','New override C; changed H and after J. Baseline class already existed in image.'),
('ContentAddressedPages.store','A fresh payload per logical slot; helper absent.','Per-group SHA256 content blobs, hardlink logical slots, last-reference cleanup.','Avoid identical physical writes without predicting future demand.','Introduced C; retained.'),
('GB10ParkingScheduler.__init__/admission/growth','A custom scheduler default32768 reservation unit.','Read parking_reservation_tokens; active setting7680.','Reduce future capacity reserved and premature spills.','New override J; underlying scheduler pre-existed.'),
('BlockPool.get_new_blocks/free_blocks/reset and eviction hooks','Normal allocator evicts hashes when reusing blocks; no new native completion hook.','Notify pressure store before overwrite; completion-indexed pages treated cached for LRU.','Keep GPU completion state evictable while preserving it before reuse.','Introduced after J in native-pressure path; revised GPU completion.'),
('NativePressureCache.before_reuse/add_metadata/consume_completions','No such helper in A/J.','Plan backing at physical reuse and coordinate ACK lifecycle.','Pressure-only native spill and ownership safety.','Introduced after J.'),
('copy_accepted_state; launch','A/J accepted state staged through older capture path.','Batched GPU byte-copy/select/shift kernel normalizes recurrent/convolution accepted state.','Avoid heavy CPU staging at every decode grid boundary.','Introduced after J.'),
('capture_gpu_completions; copy_resident_pages','A/J uses older transfer/capture machinery.','Build batched GPU capture and direct GPU missing-tail/state recall.','Share history and copy only needed state/tail on GPU.','Introduced after J.')]
inv=json.loads((p/'override-inventory.json').read_text());rows=[]
for x,v in zip(inv,info):rows.append([x['number'],x['file'],*v])
r.append(table(['#','File relative to vLLM','Function/class affected','Upstream/pre-patch or A baseline','Now','Reason','When'],rows))
r.append('''Exact active host source paths, mounted destination hashes on each rank and all affected AST symbols are in the linked inventory and [rank0 verification](override-verification-r0.json)/[rank1 verification](override-verification-r1.json). There are26 mounted Python files per container, not just22: the extra launch/initialization files are listed in section6. The manifest count is not the count of all modifications to vLLM.

## 3. Attempted changes, including rejected/reverted work

This is the complete set of change families identified in the saved six-hour ledger and subsequent deployment reports. Exact completeness against unrecorded shell actions is UNKNOWN. The original chronological [ledger](../performance-until-10am-20260914/LEDGER.md) is preserved; known erroneous test attributions there are superseded below, not deleted.

''')
attempts=[
['A baseline','Aligned64 completion shadows + deterministic MTP3','Starting point','Already active before window','Superseded','Not stock baseline.'],
['B inherited full pages','Reuse restored immutable full attention keys; exclude tail/recurrent/ring/local-only pages','Reduce repeat copy/storage','CPU invariants passed; model third pressure wave inherited3 with exact continuation','Merged H; retained concept','Standalone B not deployed; C used first.'],
['C content dedup','Per-group exact-content SHA256 blobs/hardlinks; transfer events','Avoid duplicate physical writes','Initial disk probe0reads = coverage fail; independent oracle271564800bytes read vs resident0 exact','Kept','Measured substantial logical→physical savings.'],
['D coarse tile tuning','HCdown/BA64×64 four-warps allM','Reduce small-M GEMM cost','120 within-plan cases pass; HC M1600~56% slower; cross-plan CUDA probeOOM before comparisons','Not deployed as D','Replaced by small-M-only fine plans.'],
['E packed spans','Write only active group spans, vectored short-I/O handling','Avoid padded bytes','CPU tests pass; actual full-page hashes show all22,630,400bytes covered','Rejected without restart','No demonstrated saving in actual layout.'],
['Fine tile variant','HC32×32/BA16×16, M≤128; K64 unchanged','Improve decode while preserving prefill plan','96 cross-tile/all-row cases pass; synthetic1.795×/2.678×','Kept H onward','LargeM retains original plan.'],
['F readback control / H composite','Add optional GPU readback toggle, combine B+C+fine plans','Measure verification cost','8×120sABBAABBA: full164.499t/s vs off161.718; loads7.268 vs4.801s/GiB; confounded','Control code kept; disabling reverted','Full verification restored; no aggregate benefit established.'],
['C→H workload pause','Pause remaining arm during controlled restart','Obtain usable transition despite unbounded tail','Remaining arm finished incomplete with660s tool timeout after~20min pause','Operational event, not kept setting','Transition-affected arm excluded; intervention harmed that run.'],
['I MTP5','3→5; graphs+160/192','More accepted draft tokens','Initial disk-space preflight failed~25MBshort; pip cache cleanup~937MB then retry; layout1600 auto3200 incompatible ring12','Failed/superseded','Stopped after discovered initialization failure; no successful I benchmark.'],
['I2','Physical block/retention1920','Fix MTP5 state/ring geometry','Layout fixed; existing scheduler3-draft guard rejected5','Failed/superseded','No successful I2 benchmark.'],
['I3','Guard permits MTP≤5','Run MTP5 with compatible layout','Primary/mixed and actual disk gates pass in tested routes; cold/cache still differs; quality3/8','Kept during window, later MTP3 restored','No universal determinism or isolated speedup.'],
['Fragment cleanup','Retire unreferenced remnants of incomplete completion records','Recover capacity from unusable checkpoints','CPU allocator/lookup tests passed; J only3remaining reference opportunities, reclaimable bytesUNKNOWN','Not deployed','No demonstrated gain warranting restart.'],
['J reservation','32768→7680 scheduler reservation; launcher option','Avoid premature spill from advance reservation','Growth7680/15360 passed; writes127.42→82.88GB vs I3; rate143.52→131.83','Kept','Different concurrency/workload; not isolated causal win.'],
['J growth harness correction','Read choices[0].prompt_token_ids instead of top-level','Correct invalid test assertion','Original KeyError retained; corrected fresh-salt growth passed','Harness fix kept','No serving-code change.'],
['J disk-oracle target retries','Choose actual disk-backed target pressure5-31','Establish foreground disk coverage','Two targets resident; final325877760bytes vs resident0 with exact prefix/tokens/scores','Successful oracle kept as evidence','Earlier failed coverage preserved.'],
['Write delay/group exclusion','Investigated selective admission/delay','Avoid unused writes','All groups recalled; some reads~0.52–8.10s after writes','Not deployed','Cannot infer unused future content from aggregate ratios; RAM-buffer alternative untested.'],
['Decode-aware small prefill budget','Proposed128/256 when decode active','Smooth low per-stream speed','Diagnostic found long prefills; no serving change in report','Not deployed','User prioritizes aggregate throughput; isolated benefitUNKNOWN.'],
['MTP3-only reload after J','Return speculation depth3 before other fixes','Apples-to-apples baseline','Loading stopped on user instruction','Aborted','Avoid extra baseline-only reload.'],
['Completion/suspend event candidate','Remove periodic64 capture, exact accepted event state, partial restore grid','Reduce heavy checkpoint work','40CPU state cases;1260scheduler cases;28C4 comparisons; forced disk coverage inconclusive','Deployed then superseded transport','Event mechanism retained now.'],
['Native-pressure-only candidate','Disable completion snapshots; ordinary APC, spill on allocator reuse; retention0','GPU native cache primary, no work before pressure','C10+serial equal,0disk; source7264→follow-up7396 cached3840; replay regression','Deployed then superseded','Removed too much endpoint reuse.'],
['GPU-native completion v1','Borrow GPU attention/ring; capture only recurrent state; normal LRU','Restore endpoint reuse cheaply','Review found lookup→allocation source lifetime race','Startup stopped; superseded before live validation','Not counted as correctness pass.'],
['GPU-native completion v2','Pin lookup sources through allocation/load; release unscheduled pins; ownership fixes','Close race and leave pages normally evictable','Component both-rank disk roundtrip pass;15live C4/C10/serial exact;75/80real endpoint hits','Running now','Five misses and cold/cache parity unresolved; no new full-model pressure oracle.'],
['Native finer matching audit','Test requested16/32/64 blocks and native64 hash matching without service change','Determine simpler alternative','Physical blocks auto1600; partial-tail registers input boundary only','Read-only checks; not deployed','Sol independently established stock defaults; report linked in audit addendum.']]
r.append(table(['Attempt','Change','Hypothesis','Result','Disposition','Reason/limit'],attempts))
r.append('''## 4. Unresolved correctness and verification

**Deterministic serving has NOT been achieved in the broad sense of the same request producing identical output regardless of cache route, restart, or speculation configuration.** Tested temperature-zero concurrency within particular configurations/routes passes. These claims must remain separate.

### Cold versus cached: measured difference and unknown cause

Archived J probe used identical7393-token continuation input, temperature0, seed917352, max_tokens16. Warm cached7360 tokens; cold cached0 under another cache salt. Warm C4 copies agree with warm serial. Cold and warm differ:

- Warm: `</think>\\n\\n1. Concurrent model inference refers to executing multiple inference requests, model components`
- Cold: `</think>\\n\\n1. Concurrent model inference is the execution of multiple model workloads,`
- First selected-token difference: position8 (one-based), warm token18675 ` refers`, cold token369 ` is`.
- Nine of16 token positions differ; after position8 the prefixes differ, so later score differences cannot isolate one operation.
- At position8, identical preceding token IDs: warm logprobs ` refers`=-0.5053287148, ` is`=-1.1303286552 (refers favored0.6249999404). Cold ` refers`=-0.9474591017, ` is`=-0.6974591017 (is favored0.25). Relative preference shifts0.8749999404 logit units.
- Scores already differ at position1: selected logprob warm−0.0765574500 vs cold−0.1148261949 (absolute gap0.0382687449); matching token choices do not imply matching computation.
- H and J archived responses match each other when compared **within the same route**, including reproducing this cold/warm difference. This does not validate the route difference.

**Why it happens: UNKNOWN at the responsible-operation level.** The two routes compute/reuse state differently, but there is no completed first-differing-tensor trace that proves whether residual arithmetic-path differences or checkpoint-state capture/attribution cause this failure. It must not be dismissed as a harmless score perturbation or proven correct by a disk byte checksum. Raw tokens and all top5 reported logprobs are in [cold-warm-difference.json](cold-warm-difference.json); original files in `../aggregate-throughput-20260914/after-J/`.

### Other unresolved scopes

- MTP3 vs deterministic no-spec whole model differs: first returned score at index0, first selected token at index38 in earlier report. Primitive recurrence equivalence does not prove whole-model equivalence. Cause UNKNOWN.
- Current GPU-native-completion candidate:15 live route comparisons passed, but no new cold-vs-cached equality proof; archived failure is not declared fixed. Current same-input cold/cached delta: UNKNOWN (not newly measured).
- Five of80 current sampled real follow-ups missed estimated completion endpoint. Two lookups were logged before snapshot publication; exact attribution across all five between timing and rendered-prefix changes UNKNOWN. Mean estimated endpoint replay185.75; semantic wire comparison is not raw generated-token proof.
- Current isolated GPU/disk byte roundtrips pass on both ranks; a new **full-model** forced disk-resume oracle and full-context C20 suspension/eviction validation remain pending/not performed. H/J disk oracles do not automatically qualify new ownership code.
- Exhaustive arbitrary concurrency/length/stochastic-sampling/cross-restart equivalence not established. Configured32 requests differs from observed test peaks8 or10.
- Bounded disk backing can fail to admit backing if slots pinned; cache may be recomputed. Long-run failure recovery/ownership under every interleaving not established by component tests.
- Startup numeric-format/NumPy and interface warnings persist historically; no evidence here attributes the cold/cache difference to them.

### Supplementary workload verification: exact status

The **1 pass /4 failures /3 pending** is the frozen J09:00UTC result, not a current inference-kernel test. Benchmark and original CLI checks passed for the five completed arms, but generated repairs failed the additional working-directory test in four. Correct source: `operator-audit/relative-path-cwd.json`, preserved per arm; earlier egg/zip explanations used the wrong artifact and are retracted.

''')
r.append(table(['J arm','Supplementary at09:00UTC','Failure/result'],[
['temperature-0p0-r01','PASS','Second relative lookup returns[src,project].'],['temperature-0p0-r02','FAIL','Relative-path lookup after chdir raises ImportError; absolute control succeeds.'],['temperature-0p2-r01','PENDING','Arm not completed at cutoff.'],['temperature-0p4-r01','FAIL','Same stale-CWD/relative ImportError pattern.'],['temperature-0p6-r01','FAIL','Same stale-CWD/relative ImportError pattern.'],['temperature-0p8-r01','PENDING','Arm not completed at cutoff.'],['temperature-1p0-r01','FAIL','Same stale-CWD/relative ImportError pattern.'],['temperature-1p2-r01','PENDING','Arm not completed at cutoff.']]))
r.append('''**Recovered during this audit: all three formerly pending J arms also failed. Final J supplementary status is1 pass /7 failures /0 pending.** Their actual saved stdout and returncodes are in [J-supplementary-recovered.json](J-supplementary-recovered.json). Every failure shows stale working-directory state with second relative-path ImportError and successful absolute control. The09:00 cutoff remains1/4/3; it is not the final result. H finished5/8 supplementary passes; I3 finished3/8. These are real incomplete-repair outcomes, not explained away by passing benchmark checks. Causal attribution to serving changes is UNKNOWN because trajectories and generated patches differ. Authoritative files: `J-actual-supplementary-final.json`, `quality-actual-supplementary-audits.json`, `QUALITY-CORRECTION.md` in the historical directory.

## 5. H versus I3 versus J

Same eight-arm workload definition and approximately matched elapsed observation from each launch, **not identical request streams or concurrency**. H includes the readback experiment and probes. J baseline precedes campaign creation by0.34s. Rates include client/tool waits and declining arm counts. Filesystem payload bytes are summed across ranks and are NOT NVMe block-device bytes. Prompt throughput includes cache hits and is NOT uncached prefill compute throughput.

''')
c=json.loads((p/'comparison-expanded.json').read_text())['rows'];u=json.loads((h/'H-I3-J-final-unread-matched-elapsed.json').read_text())['rows']
metrics=[('Window seconds','seconds'),('Generated tokens','generated_tokens'),('Generation tok/s','aggregate_tps'),('Completed requests','completed_requests'),('Mean sampled running','mean_running_samples'),('Accepted draft tokens/attempt','accepted_drafts_per_attempt'),('Preemptions','preemptions'),('Uncached prompt tokens','uncached_prompt_tokens'),('Written payload bytes','payload_written'),('Read payload bytes','payload_loaded')]
rows=[[name,*[round(x[k],6) for x in c]] for name,k in metrics]
for k in ['prompt_tokens_total','prompt_tokens_cached_total','prompt_tps','prefix_cache_queries_total','prefix_cache_hits_total','external_prefix_cache_queries_total','external_prefix_cache_hits_total']:rows.append([k,*[round(x['additional'][k],6) for x in c]])
rows += [['Combined prompt reuse %',*[round(x['cache_fraction']*100,6) for x in c]],['Confirmed retired-unread bytes',*[x['retired_unread_bytes'] for x in u]],['Retired-unread / written %',*[round(x['confirmed_unread_lower_bound']*100,6) for x in u]],['Logical disk bytes requested in exact matched interval','UNKNOWN','UNKNOWN','178581012480 (08:10:30–09:00 event interval; sampling end differs by~7.7s)'],['MTP depth',3,5,5],['Physical token block',1600,1920,1920],['Reservation token increment',32768,32768,7680],['Capture policy','Periodic64','Periodic64','Periodic64'],['Integrity','Checksum+readback; readback transiently off in4 windows','Checksum+readback','Checksum+readback'],['Concurrency/route verification','Passed tested routes; cold/cache FAIL','Passed tested routes; cold/cache FAIL','Passed tested routes; cold/cache FAIL'],['Supplementary','5pass/3fail','3pass/5fail','1pass/4fail/3pending at cutoff']]
r.append(table(['Metric','H','I3','J'],rows))
r.append('''Exact matched-window logical requested bytes for H/I3 are UNKNOWN in the ready-made comparison: aggregate event extracts do not carry timestamps. Do not replace them with whole-cycle totals. Raw timestamped logs may permit a separate recovery. J's event interval gives178,581,012,480 requested bytes,82,881,576,960 written,95,699,435,520 avoided by dedup (53.5888%). Its payload written/read agrees with the matched comparison, but the time intervals are explicitly distinguished.

Retired-unread is a conservative cohort lower bound: only blobs both attached and retired inside the window; excludes live unread and preexisting blobs; any reader counts. It is not a measured achievable online-policy saving. H vs J generated rate−0.99%, I3 vs J−8.14%; neither comparison isolates reservation or MTP because concurrency/context/cache trajectories differ.

The current167.08tok/s120-second8–10-request sample is from a later campaign. It must NOT be placed in this table as a matched fourth arm. Current sustained recovery to historical200+tok/s: NOT ESTABLISHED.

## 6. Exact currently running arguments, environment, and mounts

The following material comes from `docker inspect` and `/proc`, not reconstruction from deploy_config.json. Docker entrypoint argv is distinct from actual PID1 after container_entry.py has exec'd vLLM. Child engine/worker processes rewrite argv to process titles; their original spawn argv is therefore UNKNOWN from /proc, while observed titles and environments are preserved. Credential strings are redacted; unredacted credentials are not needed to audit serving behaviour.

''')
for rank in [0,1]:
 d=json.loads((p/f'live-r{rank}.json').read_text());r.append(f'### Rank {rank}\n\n```text\n'+(p/f'live-r{rank}.txt').read_text()+'```\n\n')
 r.append('All mounts, verbatim captured source/destination and read/write flag:\n\n');r.append(table(['Source','Destination','RW'],[[x['Source'],x['Destination'],x['RW']] for x in d['container']['Mounts']]))
r.append('''## Audit limits

UNKNOWN items above are explicit gaps, not inferred passes. Source-hash agreement, HTTP health, byte-exact transport, within-route deterministic sampling, and generated-code correctness are different checks. None substitutes for the others. This report preserves failed coverage, rejected candidates and unresolved output divergence alongside successful gates.
''')
(p/'AUDIT.md').write_text('\n'.join(r));print('Wrote',p/'AUDIT.md')
