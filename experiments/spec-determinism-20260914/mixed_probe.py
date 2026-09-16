"""Compare prompt-specific reference tokens and full returned scores under C8 load."""
import concurrent.futures, hashlib, json, time, urllib.request, threading
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/('mixed-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()));OUT.mkdir()
key=Path('/home/jon/.config/qwen38/api-key').read_text().strip()
base='http://127.0.0.1:30001'
source=json.loads((ROOT/'request.json').read_text())
def run(i,phase):
    payload=dict(source)
    payload.update(messages=[{'role':'user','content':('A sample sentence about concurrency and numerical arithmetic.\n' * [1,12,180,350][i])+'Explain why parallel requests can change floating point results. Give a detailed explanation.'}],max_tokens=96,seed=917352)
    payload.pop('tools',None);payload.pop('tool_choice',None)
    payload['request_id']=f'mixed-invariance-{phase}-{i}-{time.time_ns()}'
    req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    chunks=[];start=time.monotonic()
    with urllib.request.urlopen(req,timeout=240) as response:
        for line in response:
            if line.startswith(b'data: ') and line.strip()!=b'data: [DONE]':chunks.append(json.loads(line[6:]))
    choices=[c for ch in chunks for c in ch.get('choices',[])]
    row={'i':i,'phase':phase,'seconds':time.monotonic()-start,'tokens':[t for c in choices for t in c.get('token_ids',[])], 'scores':[p for c in choices for p in (c.get('logprobs') or {}).get('content',[])], 'usage':[ch['usage'] for ch in chunks if ch.get('usage')], 'prompt_ids':next((ch['prompt_token_ids'] for ch in chunks if ch.get('prompt_token_ids')),None),'chunks':chunks}
    (OUT/f'{phase}-{i}-{time.time_ns()}.json').write_text(json.dumps(row))
    assert row['tokens'] and row['scores'], 'Missing diagnostics'
    print(phase,i,len(row['tokens']),round(row['seconds'],2),flush=True)
    return row
stop_metrics=threading.Event()
def collect_metrics():
    with (OUT/'metrics.jsonl').open('w') as f:
        while not stop_metrics.is_set():
            try:
                with urllib.request.urlopen(base+'/metrics',timeout=2) as response:raw=response.read().decode()
                lines=[line for line in raw.splitlines() if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{'))]
                f.write(json.dumps({'time':time.time(),'metrics':lines})+'\n');f.flush()
            except Exception as exc:f.write(json.dumps({'error':repr(exc)})+'\n')
            stop_metrics.wait(.25)
monitor=threading.Thread(target=collect_metrics,daemon=True);monitor.start()
refs=[run(i,'reference') for i in range(4)]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    futs=[]
    for i in [0,3,1,2,0,3,1,2]:
        futs.append(pool.submit(run,i,'concurrent'));time.sleep(.15)
    rows=[f.result() for f in futs]
after=[run(i,'after') for i in range(4)]
summary=[]
for row in rows+after:
    ref=refs[row['i']]
    summary.append({'i':row['i'],'phase':row['phase'],'tokens_equal':row['tokens']==ref['tokens'],'scores_equal':row['scores']==ref['scores'],'prompt_equal':row['prompt_ids']==ref['prompt_ids'],'seconds':row['seconds']})
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

stop_metrics.set();monitor.join(timeout=3)
