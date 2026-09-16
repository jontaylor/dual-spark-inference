"""Compare native policy and aligned ABBA cycles; bootstrap cycles, not steps."""
import argparse
import collections
import csv
import json
from pathlib import Path
import random
import statistics

p=argparse.ArgumentParser();p.add_argument('captures',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--block-steps',type=int,default=64);p.add_argument('--min-phase-samples',type=int);p.add_argument('--metric',choices=['next_period_ms','gpu_hash_to_gate_ms'],default='next_period_ms');args=p.parse_args()
identities=set();policy_kinds=set()
for capture in args.captures:
 metadata_path=capture/'capture.json'
 if metadata_path.exists():
  metadata=json.loads(metadata_path.read_text())
  identities.add(tuple(sorted((r['rank'],r['pid']) for r in metadata['ranks'])))
  policy_kinds.add(metadata.get('async_policy','forced-worker'))
if len(identities)>1:raise ValueError('Cannot pool sequence counters across worker restarts')
if len(policy_kinds)>1:raise ValueError('Cannot pool different async policy implementations')
async_policy=next(iter(policy_kinds),'unspecified')
minimum=args.min_phase_samples if args.min_phase_samples is not None else args.block_steps//2
if minimum<1:raise ValueError('Minimum phase sample count must be positive')
def summary(values):
 values=sorted(values)
 if not values:return None
 return {'n':len(values),'mean':statistics.mean(values),'median':statistics.median(values),'p95':values[min(len(values)-1,int(.95*len(values)))],'max':max(values)}
def boot(values):
 if len(values)<3:return {'cycles':len(values),'mean_delta_ms':statistics.mean(values) if values else None,'interval':None}
 rng=random.Random(933);means=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(5000))
 return {'cycles':len(values),'mean_delta_ms':statistics.mean(values),'interval95_ms':[means[125],means[4874]]}
all_ranks=[];result={}
metrics=['gpu_hash_to_gate_ms','submit_plus_gpu_wait_ms','submit_ms','complete_ms','gpu_wait_us','gpu_wait_polls','next_period_ms','misses','hits']
for rank in (0,1):
 rows={}
 for capture in args.captures:
  for raw in csv.DictReader((capture/f'rank{rank}-steps.csv').open()):
   row={k:(float(v) if v and k!='policy' else v) for k,v in raw.items()}
   if isinstance(row.get('gpu_wait_us'),(float,int)) and isinstance(row.get('submit_ms'),(float,int)):
    row['submit_plus_gpu_wait_ms']=row['submit_ms']+row['gpu_wait_us']/1000
   seq=int(row['seq']);rows[seq]=row
 all_ranks.append(rows)
 groups=collections.defaultdict(list);cycles=collections.defaultdict(lambda:collections.defaultdict(list));mismatches=[]
 for seq,row in sorted(rows.items()):
  decode=bool(row['deferred']) if isinstance(row.get('deferred'),(int,float)) else row['tokens']<=4*row['requests']
  if not decode or not row.get('policy'):continue
  block=(seq-1)//args.block_steps;phase=block%4
  expected='async' if phase in (1,2) else 'inline'
  if row['policy']!=expected:mismatches.append(seq)
  key=(int(row['rows']),int(row['requests']),row['policy']);groups[key].append(row)
  # Omit switching boundary: next-period measurement may include transition work.
  if (seq-1)%args.block_steps in (0,args.block_steps-1):continue
  has_metric=isinstance(row.get(args.metric),(int,float))
  stable_next=(row.get('next_rows',row['rows'])==row['rows']
      and row.get('next_requests',row['requests'])==row['requests']
      and (bool(row['next_deferred']) if isinstance(row.get('next_deferred'),(int,float)) else row.get('next_tokens',row['tokens'])<=4*row['requests']))
  if has_metric and (args.metric!='next_period_ms' or stable_next):
   cycles[(block//4,int(row['rows']),int(row['requests']))][phase].append(row)
 grouped={f'{count}/r{requests}/{policy}':{k:summary([x[k] for x in rr if isinstance(x.get(k),(int,float))]) for k in metrics} for (count,requests,policy),rr in sorted(groups.items())}
 paired=collections.defaultdict(list);cycle_rows=[]
 for (cycle,count,requests),phases in sorted(cycles.items()):
  if set(phases)!={0,1,2,3} or min(map(len,phases.values()))<minimum:continue
  means={phase:statistics.mean(x[args.metric] for x in rr) for phase,rr in phases.items()}
  delta=(means[1]+means[2]-means[0]-means[3])/2
  paired[f'{count}/r{requests}'].append(delta);cycle_rows.append({'cycle':cycle,'rows':count,'requests':requests,'phase_n':{k:len(v) for k,v in phases.items()},'phase_means_ms':means,'async_minus_inline_ms':delta})
 result[rank]={'decode_policy_mismatches':mismatches,'groups':grouped,'paired_cycles':{count:boot(v) for count,v in paired.items()},'cycle_details':cycle_rows}
common=set(all_ranks[0])&set(all_ranks[1]);mismatch=[]
for seq in sorted(common):
 for key in ['rows','tokens','requests','policy']:
  if all_ranks[0][seq].get(key)!=all_ranks[1][seq].get(key):mismatch.append([seq,key,all_ranks[0][seq].get(key),all_ranks[1][seq].get(key)])
result['alignment']={'common_sequences':len(common),'mismatches':mismatch}
result['async_policy']=async_policy
result['minimum_phase_samples']=minimum
result['paired_metric']=args.metric
result['interpretation']='Negative async-minus-inline delta favors async. Confidence interval resamples whole complete ABBA cycles; dynamic workload still limits causal attribution. GPU wait is kernel-body time. Submit-plus-GPU-wait is a phase-cost proxy, not a measured end-to-end effect or exact TP critical path. Stratify rows; never pool restart sequence counters.'
args.output.write_text(json.dumps(result,indent=2)+'\n')
for rank in (0,1):
 print('rank',rank,'paired',result[rank]['paired_cycles'],'policy mismatches',len(result[rank]['decode_policy_mismatches']))
 for key,d in result[rank]['groups'].items():
  if d['submit_ms'] and d['submit_ms']['n']>=20:print(key,{k:({f:v for f,v in d[k].items() if f in ['n','median','p95','max']} if d[k] else None) for k in ['submit_ms','gpu_wait_us','next_period_ms']})
print('alignment',result['alignment'])
