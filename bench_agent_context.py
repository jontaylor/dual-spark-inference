#!/usr/bin/env python3
"""Long-context agent-code benchmark. Count accepted output, never prompt tokens.

Example: venv/bin/python bench_agent_context.py --context 150000 --concurrency 1,2,4,8,12,16,20,24
Context is the total prompt+output allowance. Prefix reuse is deliberately
prevented by a distinct initial session marker for each request.
"""
import argparse
import asyncio
import hashlib
import json
from runtime_config import load_config, model_snapshot, resolve_path
import statistics
import time
from pathlib import Path

import aiohttp
from tokenizers import Tokenizer

parser=argparse.ArgumentParser()
parser.add_argument('--context',type=int,default=8192)
parser.add_argument('--output',type=int,default=256)
parser.add_argument('--concurrency',default='1,2,4,8,12,16,20,24')
parser.add_argument('--repeats',type=int,default=1)
parser.add_argument('--repeat-prefixes',action='store_true',help='Reuse each unique prompt on later repeats to measure actual prefix-cache retention')
parser.add_argument('--warm-prefixes',action='store_true',help='Prefill each unique agent context before the timed decode run')
parser.add_argument('--base',default='http://127.0.0.1:30001')
parser.add_argument('--corpus',required=True)
parser.add_argument('--results',default='./results')
args=parser.parse_args()
root=Path(__file__).resolve().parent
cfg=load_config(root)
snapshot=(resolve_path(cfg['paths']['hf_cache'], root) / ('models--' + cfg['model_id'].replace('/', '--')) / 'snapshots')/cfg['revision']
tokenizer=Tokenizer.from_file(str(snapshot/'tokenizer.json'))
documents=[]
for p in sorted(Path(args.corpus).rglob('*')):
    if p.suffix in ('.cpp','.h','.c','.py') and p.is_file() and p.stat().st_size<2_000_000:
        documents.append(f'\n<file path="{p.relative_to(args.corpus)}">\n{p.read_text(errors="replace")}\n</file>\n')
corpus=''.join(documents)
corpus_ids=tokenizer.encode(corpus,add_special_tokens=False).ids
if len(corpus_ids)<args.context:
    raise RuntimeError(f'Need more real source material: {len(corpus_ids)} tokens available')
result_dir=Path(args.results);result_dir.mkdir(parents=True,exist_ok=True)
run=str(int(time.time()))
key=resolve_path(cfg['api_key_file'], root).read_text().strip()
headers={'Authorization':'Bearer '+key}

async def prepare(session,ident):
    marker='WREN-'+hashlib.sha256(ident.encode()).hexdigest()[:12]
    prefix=f'Session {ident}. You are reviewing a repository for a coding agent.\n'
    suffix=(f'\nReview request: identify three concrete ownership, concurrency, or error-handling risks in the supplied code. '
            f'Cite file and function names and explain how to investigate each. Do not invent executed tests. '
            'Find the RUNBOOK_VERIFICATION marker embedded in the excerpt and begin your response with its value.\n')
    # Leave allowance for chat-template overhead, then use the server tokenizer
    # to trim to the exact target without cutting template/control tokens.
    budget=args.context-args.output
    n=budget-256
    for _ in range(3):
        body=(tokenizer.decode(corpus_ids[:n//2],skip_special_tokens=False)+
              f'\nRUNBOOK_VERIFICATION={marker}\n'+
              tokenizer.decode(corpus_ids[n//2:n],skip_special_tokens=False))
        content=prefix+body+suffix
        messages=[{'role':'user','content':content}]
        payload={'model':'qwen3.8-flash-next','messages':messages,'add_generation_prompt':True,
                 'chat_template_kwargs':{'enable_thinking':False}}
        async with session.post(args.base+'/tokenize',json=payload) as response:
            response.raise_for_status();data=await response.json()
        count=data.get('count',len(data.get('tokens',[])))
        if count<=budget and budget-count<8:break
        n+=budget-count
    if count>budget:raise ValueError(f'Prompt exceeds budget: {count}>{budget}')
    return messages,marker,count

async def request(session,prepared):
    messages,marker,count=prepared
    payload={'model':'qwen3.8-flash-next','messages':messages,'max_tokens':args.output,'temperature':0,
             'stream':True,'stream_options':{'include_usage':True,'continuous_usage_stats':True},
             'chat_template_kwargs':{'enable_thinking':False}}
    wall_start=time.time();start=time.monotonic();first=None;usage=None;text='';finish=None;timeline=[];request_metrics=None
    async with session.post(args.base+'/v1/chat/completions',json=payload) as response:
        response.raise_for_status()
        async for raw in response.content:
            line=raw.decode().strip()
            if not line.startswith('data: ') or line=='data: [DONE]':continue
            data=json.loads(line[6:])
            if data.get('metrics'):request_metrics=data['metrics']
            if data.get('usage'):
                usage=data['usage']
                timeline.append([time.monotonic(),usage['completion_tokens']])
            for choice in data.get('choices',[]):
                delta=choice.get('delta',{})
                piece=delta.get('content') or delta.get('reasoning_content') or delta.get('reasoning') or ''
                if piece and first is None:first=time.monotonic()
                text+=piece
                if choice.get('finish_reason'):finish=choice['finish_reason']
    end=time.monotonic()
    if not usage:raise RuntimeError('No authoritative completion-token usage in stream')
    return {'wall_start':wall_start,'start':start,'first':first,'end':end,'elapsed':end-start,
            'ttft':None if first is None else first-start,'usage':usage,'prepared_prompt_tokens':count,
            'metrics':request_metrics,'finish_reason':finish,'marker_found':marker in text,'text':text,'token_timeline':timeline}

async def main():
    timeout=aiohttp.ClientTimeout(total=14400,sock_read=7200)
    async with aiohttp.ClientSession(headers=headers,timeout=timeout,connector=aiohttp.TCPConnector(limit=64)) as session:
        async def metrics():
            async with session.get(args.base+'/metrics') as response:
                response.raise_for_status();text=await response.text()
            return [l for l in text.splitlines() if l.startswith(('vllm:num_preemptions_total{',
                'vllm:prefix_cache_hits_total{','vllm:prefix_cache_queries_total{',
                'vllm:kv_cache_usage_perc{','vllm:num_requests_running{'))]
        for c in map(int,args.concurrency.split(',')):
            for repeat in range(args.repeats):
                print(f'Preparing C{c} context={args.context} repeat={repeat}',flush=True)
                prepared=[]
                prefix_repeat = 0 if args.repeat_prefixes else repeat
                for i in range(c):prepared.append(await prepare(session,f'{run}-{c}-{prefix_repeat}-{i}'))
                warmup_start=time.monotonic()
                if args.warm_prefixes:
                    for i,(messages,_,_) in enumerate(prepared):
                        payload={'model':'qwen3.8-flash-next','messages':messages,'max_tokens':1,'temperature':0,
                                 'chat_template_kwargs':{'enable_thinking':False}}
                        async with session.post(args.base+'/v1/chat/completions',json=payload) as response:
                            response.raise_for_status();await response.json()
                        print(f'Prefilled {i+1}/{c} unique contexts',flush=True)
                warmup_seconds=time.monotonic()-warmup_start
                metrics_before=await metrics()
                # Preparation is excluded from timed requests. Prefill is included.
                rows=await asyncio.gather(*(request(session,p) for p in prepared))
                start=min(r['start'] for r in rows);end=max(r['end'] for r in rows)
                generated=sum(r['usage']['completion_tokens'] for r in rows)
                # Only report a sustained all-stream decode rate when every
                # request has started output and none has finished yet.
                common_start=max(r['first'] for r in rows)
                common_end=min(r['token_timeline'][-1][0] for r in rows)
                def count_at(row,t):
                    return max((n for stamp,n in row['token_timeline'] if stamp<=t),default=0)
                common_tokens=sum(count_at(r,common_end)-count_at(r,common_start) for r in rows)
                common_seconds=max(0,common_end-common_start)
                summary={'concurrency':c,'context_allowance':args.context,'repeat':repeat,
                    'prefix_state':('warmed_unique_contexts' if args.warm_prefixes else
                                    'reused_unique_contexts' if args.repeat_prefixes and repeat else 'cold_unique_contexts'),
                    'prefill_warmup_seconds':warmup_seconds,
                    'generated_tokens':generated,'elapsed_seconds':end-start,
                    'end_to_end_generated_tokens_per_second':generated/(end-start),
                    'common_decode_window_seconds':common_seconds,
                    'common_decode_generated_tokens_per_second':common_tokens/common_seconds if common_seconds>=5 else None,
                    'ttft_median_seconds':statistics.median(r['ttft'] for r in rows),
                    'marker_checks_passed':sum(r['marker_found'] for r in rows),'requests':rows,
                    'metrics_before':metrics_before,'metrics_after':await metrics(),
                    'config':cfg,'corpus_sha256':hashlib.sha256(corpus.encode()).hexdigest()}
                path=result_dir/f'{run}-ctx{args.context}-c{c}-r{repeat}.json'
                path.write_text(json.dumps(summary,indent=2)+'\n')
                print(json.dumps({k:v for k,v in summary.items() if k not in ('requests','config')}),flush=True)

asyncio.run(main())
