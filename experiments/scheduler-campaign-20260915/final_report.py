import pathlib,json,itertools,statistics
P=pathlib.Path(__file__).resolve().parent
rows=json.loads((P/'results.json').read_text());by={tuple(json.loads((P/r['arm']/'config-r0.json').read_text())[k] for k in ('max_num_seqs','max_num_batched_tokens','long_prefill_token_threshold')):r for r in rows}
lines=['# Scheduler campaign comparison','',f'{len(rows)}/8 measured configurations available. Each arm uses a fresh campaign and a600-second window; counter rates span the300 samples (approximately598seconds).','', '| Sequences | Budget | Threshold | Gen tok/s | Gen tok/s final5m | Mean TTFT s | Prompt reuse % | Completed requests | GPU power W combined |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for cfg,r in by.items():
 f=r['full'];power=sum(v['mean_power_w'] for v in r['gpu_half_second'].values());lines.append(f"| {cfg[0]} | {cfg[1]} | {cfg[2]} | {f['generation_tok_s']:.2f} | {r['last_five']['generation_tok_s']:.2f} | {f['mean_ttft_s']:.2f} | {100*f['cached_prompt_fraction']:.2f} | {f['deltas'].get('vllm:request_success_total',0):.0f} | {power:.2f} |")
lines+=['','## Matched configuration pairs','', 'Each row changes only the named configuration field. These are observations from single runs, not estimates with statistical confidence. Model/tool trajectories, prompt work and cache reuse can diverge even with the same frozen workload.','', '| Change | Other settings | Generation throughput change | Mean TTFT change |','|---|---|---:|---:|']
pairs=[]
for dim,(name,lo,hi) in enumerate([('Sequences',24,32),('Budget',4096,16384),('Threshold',512,2048)]):
 for cfg,a in by.items():
  if cfg[dim]!=lo:continue
  other=list(cfg);other[dim]=hi;other=tuple(other)
  if other not in by:continue
  b=by[other];gain=100*(b['full']['generation_tok_s']/a['full']['generation_tok_s']-1);lat=b['full']['mean_ttft_s']-a['full']['mean_ttft_s'];detail=', '.join(f'{label}={cfg[i]}' for i,label in enumerate(('seq','budget','threshold')) if i!=dim)
  pairs.append({'variable':name,'from':cfg,'to':other,'generation_percent_change':gain,'mean_ttft_seconds_change':lat});lines.append(f'| {name} {lo} → {hi} | {detail} | {gain:+.2f}% | {lat:+.2f}s |')
lines+=['','## Prompt work and reported transfers','', '| Arm | Local compute tokens | Local-cache tokens | External-path tokens | Reported GPU→CPU bytes | Reported CPU→GPU bytes |','|---|---:|---:|---:|---:|---:|']
for r in rows:
 d=r['full']['label_deltas']
 def match(metric,label):return sum(v for k,v in d.items() if k.startswith('vllm:'+metric+'{') and label in k)
 vals=[match('prompt_tokens_by_source_total',f'source="{v}"') for v in ('local_compute','local_cache_hit','external_kv_transfer')]+[match('kv_offload_total_bytes_total',f'transfer_type="{v}"') for v in ('GPU_to_CPU','CPU_to_GPU')]
 lines.append('| '+r['arm']+' | '+' | '.join(f'{v:.0f}' for v in vals)+' |')
lines+=['','External-path reuse includes the custom GPU-resident completion cache. These counters must not be interpreted as physical disk I/O. Host diskstats and raw connector logs are retained for further attribution.','', '## Interpretation limits','', '- No repeat runs or ninth drift-control run; the eight combinations were tested in the agreed order. Filesystem PLE cache was not flushed.','- All serving settings other than the three matrix fields were preserved. The existing fair-prefill and cache/recurrent alignment policies can reduce actual chunks below the threshold. Sampled chunk histograms are in results.json; they are not a complete per-step trace.','- Readiness/campaign reset and collection audit is in audit.json. Sampling errors in nonblocking Python profiles are preserved in the profile logs; these are not GPU kernel traces.','- Generated tokens include partial responses at cutoff. Mean latency includes observations recorded during the window, not an identical request cohort. Campaign run states and calls/tool events are separately recorded. No task-quality or deterministic-output conclusion is established.','- Arm6 had one disk-reserve preflight failure before measurement; it was recovered without changing serving settings. See RECOVERY.md. Full historical archives were relocated to the worker with hash verification; paths are in archive-relocations.json.','- The final matrix configuration stays running; the campaign does not automatically deploy whichever configuration has the largest observed throughput.']
(P/'COMPARISON.md').write_text('\n'.join(lines)+'\n');(P/'pairs.json').write_text(json.dumps(pairs,indent=2));print(f'Wrote comparison for {len(rows)} arms and {len(pairs)} pairs')
