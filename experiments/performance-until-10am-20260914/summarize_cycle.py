"""Timestamp-bounded whole-cycle counters with explicit completion/config scope."""
import argparse,datetime,json,subprocess,time
from pathlib import Path
p=Path(__file__).resolve().parent;parser=argparse.ArgumentParser();parser.add_argument('cycle');args=parser.parse_args();assert args.cycle.startswith('batch-') and args.cycle[6:].isdigit()
root='/home/jon/artifact-agent-orchestration/.artifact-agent/baseline-comparison/20260914-temperature-until-1000/'+args.cycle
code=f"import json;from pathlib import Path;p=Path({root!r});print(json.dumps({{'launch':json.loads((p/'launch.json').read_text()),'campaign':json.loads((p/'campaign.json').read_text())}}))"
remote=json.loads(subprocess.check_output(['ssh','jon@192.168.0.167','python3 -'],input=code,text=True,timeout=20));launch=datetime.datetime.fromisoformat(remote['launch']['utc']).timestamp();finished=remote['campaign'].get('finished_utc');finish=datetime.datetime.fromisoformat(finished).timestamp() if finished else None
rows=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()];valid=[r for r in rows if r.get('metrics')];base=max((r for r in valid if r['time']<=launch),key=lambda r:r['time']);end=min((r for r in valid if r['time']>=finish),key=lambda r:r['time']) if finish and any(r['time']>=finish for r in valid) else valid[-1]
assert launch-base['time']<30,'Missing start counters';epochs={r['config_sha256'] for r in valid if base['time']<=r['time']<=end['time']};assert len(epochs)==1,'Cycle crosses configurations; split before comparison'
delta={k:v-base['metrics'].get(k,0) for k,v in end['metrics'].items()}
def total(n):return sum(v for k,v in delta.items() if k.startswith('vllm:'+n+'{'))
seconds=end['time']-base['time'];prompt=total('prompt_tokens_total');out={'cycle':args.cycle,'launch_utc':remote['launch']['utc'],'finished_utc':finished,'campaign_status':remote['campaign']['status'],'complete_counter_coverage':bool(finish and end['time']>=finish),'counter_start':base['time'],'counter_end':end['time'],'start_offset_seconds':launch-base['time'],'end_offset_seconds':end['time']-finish if finish else None,'seconds':seconds,'config_sha256':epochs.pop(),'generated_tokens':total('generation_tokens_total'),'aggregate_generated_tps':total('generation_tokens_total')/seconds,'prompt_tokens':prompt,'cached_tokens':total('prompt_tokens_cached_total'),'cache_fraction':total('prompt_tokens_cached_total')/prompt if prompt else None,'completed_requests':total('request_success_total'),'preemptions':total('num_preemptions_total'),'disk_payload_written':total('kv_offload_store_bytes_total'),'disk_payload_loaded':total('kv_offload_load_bytes_total'),'scope':'Whole sample-defined cycle interval including client/tool waits and declining active-arm count. Counter offsets reported. Filesystem payload, not device I/O. No cross-configuration speedup implied.'}
assert min(out[n] for n in ('generated_tokens','prompt_tokens','disk_payload_written','disk_payload_loaded'))>=0,'Counter reset detected'
control_file=p/'readback-paired-H/controls.jsonl'
controls=[json.loads(line) for line in control_file.read_text().splitlines()] if control_file.exists() else []
prior=[c for c in controls if c['confirmed']<=base['time']]
inside=[c for c in controls if base['time']<c['confirmed']<=end['time']]
initial=prior[-1]['gpu_readback'] if prior else True
out['gpu_readback_initial']=initial;out['gpu_readback_control_events']=inside
out['mixed_verification_modes']=any(c['gpu_readback']!=initial for c in inside)
out['verification_note']='Hot verification changes do not alter deploy_config hash; use explicit control events and paired phase windows.'
(p/(args.cycle+'-counters.json')).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
