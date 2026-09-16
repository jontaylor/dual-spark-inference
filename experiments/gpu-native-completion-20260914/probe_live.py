"""Live C1/C4/C10 token/logprob oracle and concurrent multi-turn throughput."""
import concurrent.futures
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

p=Path(__file__).resolve().parent
mode=sys.argv[1] if len(sys.argv)>1 else 'correctness'
out=p/(mode+'-'+str(time.time_ns()));out.mkdir()
key=Path('/home/jon/.config/qwen38/api-key').read_text().strip()
base='http://127.0.0.1:30001'
prompt=json.loads((p.parent/'aggregate-throughput-20260914/after-J/first.json').read_text())['response']['prompt_token_ids']

def metrics():
    s=urllib.request.urlopen(base+'/metrics',timeout=15).read().decode()
    names=['kv_completion_resident_blocks','kv_offload_store_bytes_total','kv_offload_load_bytes_total',
           'prefix_cache_queries_total','prefix_cache_hits_total','external_prefix_cache_hits_total',
           'kv_cache_usage_perc','num_requests_running','generation_tokens_total','num_preemptions_total']
    return {n:sum(float(v) for v in re.findall(r'^vllm:'+n+r'(?:\{[^\n]*\})? ([\d.eE+-]+)$',s,re.M)) for n in names}

def req(name,tokens,salt,max_tokens=64,logprobs=5):
    payload=dict(model='qwen3.8-flash-next',prompt=tokens,temperature=0,seed=917352,
                 ignore_eos=True,return_token_ids=True,cache_salt=salt,max_tokens=max_tokens,
                 logprobs=logprobs,request_id='gpu-native-'+out.name+'-'+name)
    r=urllib.request.Request(base+'/v1/completions',data=json.dumps(payload).encode(),
                            headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    start=time.time()
    with urllib.request.urlopen(r,timeout=600) as f:response=json.load(f)
    elapsed=time.time()-start
    record={'request':payload,'response':response,'start':start,'seconds':elapsed}
    (out/(name+'.json')).write_text(json.dumps(record))
    print(name,round(elapsed,3),response['usage'],flush=True)
    return record

before=metrics();(out/'before.json').write_text(json.dumps(before))
if mode=='correctness':
    salt=out.name
    seed=req('seed',prompt,salt)
    emitted=seed['response']['choices'][0]['token_ids']
    follow=prompt+emitted+[198]*100
    expected=len(prompt)+len(emitted)-1
    reference=req('c1-before',follow,salt)['response']
    rows=[]
    for concurrency in (4,10):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            rows.extend(pool.map(lambda i:req(f'c{concurrency}-{i}',follow,salt)['response'],range(concurrency)))
    rows.append(req('c1-after',follow,salt)['response'])
    checks=[{'tokens_equal':r['choices'][0]['token_ids']==reference['choices'][0]['token_ids'],
             'scores_equal':r['choices'][0]['logprobs']==reference['choices'][0]['logprobs'],
             'cached':r['usage']['prompt_tokens_details']['cached_tokens']} for r in rows]
    actual=reference['usage']['prompt_tokens_details']['cached_tokens']
    summary={'passed':actual==expected and all(c['tokens_equal'] and c['scores_equal'] and c['cached']==expected for c in checks),
             'expected_completion_boundary':expected,'reference_cached':actual,'checks':checks,
             'scope':'Background workload preserved; exact prompt-token continuation, no tool execution.'}
else:
    concurrency=int(sys.argv[2]) if len(sys.argv)>2 else 8
    rounds=int(sys.argv[3]) if len(sys.argv)>3 else 3
    output_limit=int(sys.argv[4]) if len(sys.argv)>4 else 768
    benchmark_prompts=json.loads((p/'benchmark-prompts.json').read_text())
    def chain(i):
        tokens=list(benchmark_prompts[i % len(benchmark_prompts)]['tokens']);rows=[]
        for turn in range(rounds):
            row=req(f'chain{i}-turn{turn}',tokens,out.name+f'-chain{i}',output_limit,None)
            usage=row['response']['usage'];generated=row['response']['choices'][0]['token_ids']
            expected=0 if not rows else len(tokens)-900-1
            rows.append({'turn':turn,'seconds':row['seconds'],'prompt':usage['prompt_tokens'],
                         'generated':usage['completion_tokens'],'cached':usage['prompt_tokens_details']['cached_tokens'],
                         'expected_previous_endpoint':expected})
            tokens=tokens+generated+[198]*900
        return rows
    start=time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        chains=list(pool.map(chain,range(concurrency)))
    elapsed=time.time()-start;rows=[r for chain in chains for r in chain]
    followups=[r for r in rows if r['turn']>0]
    summary={'concurrency':concurrency,'rounds':rounds,'seconds':elapsed,
             'generated':sum(r['generated'] for r in rows),
             'aggregate_output_tps':sum(r['generated'] for r in rows)/elapsed,
             'followup_mean_uncached':sum(r['prompt']-r['cached'] for r in followups)/len(followups),
             'followup_exact_endpoint_hits':sum(r['cached']==r['expected_previous_endpoint'] for r in followups),
             'followups':len(followups),'chains':chains,
             'prompt_sources':[{'arm':r['arm'],'current_tokens':len(r['tokens']),
                               'original_tokens':r['original_prompt_tokens']} for r in benchmark_prompts],
             'scope':'Synthetic multi-turn concurrent workload from diverse historical messages rendered by /tokenize: '
                     'fixed output cap, 900 newline tokens per follow-up, background preserved. '
                     'Current /tokenize lengths differ from historical chat request counts; this is not an exact historical replay. '
                     'Aggregate rate includes cold initial prefill and request overhead; not planner task completion rate.'}
after=metrics();summary.update(before=before,after=after)
summary['global_disk_store_bytes_delta']=after['kv_offload_store_bytes_total']-before['kv_offload_store_bytes_total']
summary['global_disk_load_bytes_delta']=after['kv_offload_load_bytes_total']-before['kv_offload_load_bytes_total']
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2),flush=True)
if mode=='correctness':assert summary['passed']
