import json
import sys
from pathlib import Path
from collections import defaultdict
p=Path(__file__).resolve().parent
q=p.parent/'performance-until-10am-20260914'
old=json.loads((p/'yesterday-temperature-counters.json').read_text())['rows']
if '--ten-second' in sys.argv: old=old[::2]
new=[json.loads(s) for s in (q/'metrics.jsonl').read_text().splitlines()]
release=json.loads((q/'J-release-requested.json').read_text())
raw=(q/'J-pre-resume-metrics.txt').read_text()
new.append({'time':release['time'],'metrics':{line.rsplit(' ',1)[0]:float(line.rsplit(' ',1)[1]) for line in raw.splitlines() if line.startswith('vllm:') and '_bucket{' not in line and 'config_info{' not in line}})
def val(r,n):
    if n in r['metrics']: return r['metrics'][n]
    return sum(v for k,v in r['metrics'].items() if k.startswith('vllm:'+n+'{'))
def band(n):
    if n==0:return '0'
    if n<=4:return '1-4'
    if n<=8:return '5-8'
    if n<=12:return '9-12'
    if n<=16:return '13-16'
    return '17+'
wins={'Earlier':(old,1789323720.419179,1789326681.460408),'H':(new,1789362361.6330469,1789365314.3519819),'I3':(new,1789368055.1515002,1789371007.8676856),'J':(new,1789373430.0261486,1789376392.3166676)}
out={}
for label,(source,start,end) in wins.items():
    rows=sorted([r for r in source if start<=r['time']<=end and r.get('metrics')],key=lambda r:r['time'])
    buckets=defaultdict(lambda:defaultdict(float)); exact=defaultdict(lambda:defaultdict(float)); intervals=[]
    weighted=0
    for a,b in zip(rows,rows[1:]):
        dt=b['time']-a['time']; na=val(a,'num_requests_running'); nb=val(b,'num_requests_running'); tok=val(b,'generation_tokens_total')-val(a,'generation_tokens_total')
        assert dt>0 and tok>=0
        weighted+=dt*(na+nb)/2
        for n in [na,nb]: buckets[band(n)]['endpoint_weighted_seconds']+=dt/2
        if band(na)==band(nb):
            z=buckets[band(na)];z['same_band_seconds']+=dt;z['same_band_tokens']+=tok;z['same_band_intervals']+=1
        if na==nb:
            z=exact[str(int(na))];z['seconds']+=dt;z['tokens']+=tok;z['intervals']+=1
        intervals.append({'dt':dt,'a':na,'b':nb,'tokens':tok})
    seconds=rows[-1]['time']-rows[0]['time']
    for z in buckets.values():
        z['occupancy_pct']=100*z['endpoint_weighted_seconds']/seconds
        z['same_band_tps']=z['same_band_tokens']/z['same_band_seconds'] if z['same_band_seconds'] else None
    for z in exact.values():z['tps']=z['tokens']/z['seconds']
    out[label]={'seconds':seconds,'aggregate_tps':sum(i['tokens'] for i in intervals)/seconds,'time_weighted_running':weighted/seconds,'bands':dict(buckets),'stable_endpoint_counts':dict(exact),'max_interval_seconds':max(i['dt'] for i in intervals)}
# Reweight historical stable-endpoint rates onto each later distribution, common exact counts only.
for label in ['H','I3','J']:
    earlier=out['Earlier']['stable_endpoint_counts']; later=out[label]['stable_endpoint_counts']
    shared=[n for n in earlier if n in later and earlier[n]['seconds']>=30 and later[n]['seconds']>=30]
    duration=sum(later[n]['seconds'] for n in shared)
    out[label]['standardization']={'counts':sorted(map(int,shared)),'later_seconds':duration,'later_window_coverage_pct':100*duration/out[label]['seconds'],'earlier_tps_reweighted_to_later_counts':sum(earlier[n]['tps']*later[n]['seconds'] for n in shared)/duration,'later_tps_on_same_intervals':sum(later[n]['tokens'] for n in shared)/duration}
(p/('concurrency-analysis-ten-second.json' if '--ten-second' in sys.argv else 'concurrency-analysis.json')).write_text(json.dumps(out,indent=2))
for label,d in out.items():
    print(label, 'TPS',round(d['aggregate_tps'],1),'running',round(d['time_weighted_running'],2))
    for k,z in sorted(d['bands'].items()):print(k,'occupancy',round(z['occupancy_pct'],1),'tps',round(z['same_band_tps'] or 0,1),'seconds',round(z['same_band_seconds']))
    print('standardization',d.get('standardization'))
    print('stable counts', {k:(round(z['seconds']),round(z['tps'],1)) for k,z in sorted(d['stable_endpoint_counts'].items(),key=lambda x:int(x[0]))})
