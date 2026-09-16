"""Summarize observed disk payload and logical-slot lifecycles; no model requests."""
import argparse,json,re,subprocess,shlex
from pathlib import Path
p=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--since',default='2026-09-14T03:05:05Z');parser.add_argument('--until');parser.add_argument('--label',default='content-validation');args=parser.parse_args()
reports=[]
for rank in [0,1]:
 cmd=['docker','logs','--since',args.since,f'qwen38-kv-paging-r{rank}']
 if args.until:cmd[2:2]=['--until',args.until]
 if rank:cmd=['ssh','192.168.100.11',shlex.join(cmd)]
 result=subprocess.run(cmd,capture_output=True,text=True,check=True,timeout=30);lines=(result.stdout+result.stderr).splitlines();events=[]
 for line in lines:
  if 'GB10_DISK_TRANSFER {' in line:events.append(json.loads(line.split('GB10_DISK_TRANSFER ',1)[1]))
 (p/f'{args.label}-disk-events-r{rank}.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
 states={};closed=[];groups={};unattributed_reads=0
 for e in events:
  for part in e['parts']:
   payload_size=part.get('payload_bytes_per_slot',e['page_bytes'])
   group=part['group'];g=groups.setdefault(group,{'logical_store_bytes':0,'new_payload_bytes':0,'load_bytes':0,'store_slots':0,'load_slots':0})
   for slot in part['slots']:
    if e['store']:
     if slot in states:closed.append(states.pop(slot))
     states[slot]={'group':group,'bytes':e['page_bytes'],'reads':0}
     g['logical_store_bytes']+=e['page_bytes'];g['store_slots']+=1
     if slot in part.get('new_payload_slots',part['slots']):g['new_payload_bytes']+=payload_size
    else:
     g['load_bytes']+=payload_size;g['load_slots']+=1
     if slot in states:states[slot]['reads']+=1
     else:unattributed_reads+=1
 def totals(rows):return {'logical_generations':len(rows),'logical_bytes':sum(x['bytes'] for x in rows)}
 allstates=closed+list(states.values());logical=sum(e.get('logical_bytes',e['bytes']) for e in events if e['store']);written=sum(e['bytes'] for e in events if e['store'])
 packed_logical=sum(len(part['slots'])*part.get('payload_bytes_per_slot',e['page_bytes']) for e in events if e['store'] for part in e['parts'])
 reports.append({'packing_avoided_bytes':logical-packed_logical,'content_dedup_avoided_bytes':packed_logical-written,'rank':rank,'since':args.since,'events':len(events),'logical_store_bytes':logical,'new_payload_bytes':written,'avoided_payload_bytes':logical-written,'avoided_fraction':(logical-written)/logical if logical else None,'load_bytes':sum(e['bytes'] for e in events if not e['store']),'groups':groups,'recalled_generations':totals([x for x in allstates if x['reads']]),'overwritten_without_observed_recall':totals([x for x in closed if not x['reads']]),'not_recalled_not_observed_overwritten':totals([x for x in states.values() if not x['reads']]),'reads_without_store_in_observation':unattributed_reads})
summary={'since':args.since,'until':args.until,'scope':'Filesystem payload bytes, not block-device I/O. Slot generations are logical saves; deduplicated slots may share physical content. No recall yet does not establish waste. Logical evictions without slot reuse are not observed.','ranks':reports}
(p/f'{args.label}-disk-summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
