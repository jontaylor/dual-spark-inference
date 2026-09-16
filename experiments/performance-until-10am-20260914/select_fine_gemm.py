"""Prepare a candidate only from complete, exact cross-tile benchmark evidence."""
import hashlib,json,math
from pathlib import Path
p=Path(__file__).resolve().parent
BASE=(128,128,8,3)
COUNTS={1,4,8,32,48,64,128,1600}
DECODE_M=(1,4,8,32,48,64,128)

def select(rows):
 plans={};details={}
 for name in ('hc_down','gdn_ba'):
  grouped={}
  for row in rows:
   if row['shape']==name:grouped.setdefault(tuple(row['config']),[]).append(row)
  valid={}
  for cfg,records in grouped.items():
   if len(records)!=len(COUNTS) or {r.get('M') for r in records}!=COUNTS:continue
   if not all(r.get('eligible') and r.get('fixed_k')==64 and r.get('batch_exact') and r.get('cross_config_C1_exact') and r.get('cross_config_all_rows_exact') and math.isfinite(r['milliseconds']) and r['milliseconds']>0 for r in records):continue
   valid[cfg]={r['M']:r['milliseconds'] for r in records}
  if BASE not in valid:raise RuntimeError(f'{name}: missing complete exact baseline evidence')
  # Select a decode plan only if no sampled small batch regresses more than5%.
  # Retain baseline for M>128, where small tiles can materially hurt prefill.
  candidates={cfg:math.exp(sum(math.log(valid[BASE][m]/times[m]) for m in DECODE_M)/len(DECODE_M)) for cfg,times in valid.items() if all(times[m]<=valid[BASE][m]*1.05 for m in DECODE_M)}
  best=max(candidates,key=candidates.get)
  if candidates[best]<1.05:best=BASE
  plans[name]=best;details[name]={'config':best,'decode_geomean_speedup':candidates[best],'baseline_ms':valid[BASE],'candidate_ms':valid[best],'scope':'M<=128 only; M>128 retains baseline. Synthetic kernel speedup, full-model validation required.'}
 return plans,details

def main():
 evidence=p/'gemm-fine-tuning.json';plans,details=select(json.loads(evidence.read_text()))
 original=p.parent/'targeted-determinism-20260913/targeted_kernels.dense.py';s=original.read_text().replace('# No M-indexed tuning: correctness baseline shared by prefill and decode.', '# Prefill baseline; decode tiles gated by cross-tile equality evidence.')
 grid='    grid = (min(sms, triton.cdiv(m, 128) * triton.cdiv(n, 128)),)'
 assert s.count(grid)==1 and s.count('        **GEMM_CONFIG)')==1
 selected={(336,10240):plans['hc_down'],(48,2560):plans['gdn_ba']}
 replacement=f'''    # Cross-tile equality gated by complete fine-GEMM probe; K64 throughout.
    plans = {selected!r}
    bm, bn, warps, stages = plans.get((n, k), {BASE!r}) if m <= 128 else {BASE!r}
    config = dict(BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=64,
                  GROUP_SIZE_M=8, num_stages=stages, num_warps=warps)
    grid = (min(sms, triton.cdiv(m, bm) * triton.cdiv(n, bn)),)'''
 s=s.replace(grid,replacement).replace('        **GEMM_CONFIG)','        **config)')
 target=p/'targeted_kernels.fine.py';compile(s,str(target),'exec');target.write_text(s)
 for rank in (0,1):
  cfg=json.loads((p/f'candidate-content-r{rank}.json').read_text())
  key='model_executor/determinism/gb10_targeted.py';cfg['runtime_overrides'][key]=str(target);cfg['runtime_override_sha256'][key]=hashlib.sha256(target.read_bytes()).hexdigest()
  (p/f'candidate-fine-r{rank}.json').write_text(json.dumps(cfg,indent=2)+'\n')
 details['evidence_sha256']=hashlib.sha256(evidence.read_bytes()).hexdigest();(p/'gemm-fine-selection.json').write_text(json.dumps(details,indent=2));print(json.dumps(details,indent=2))
if __name__=='__main__':main()
