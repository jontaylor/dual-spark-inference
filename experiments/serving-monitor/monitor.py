"""Read-only serving monitor: Prometheus, existing accounting logs, optional PLE uprobes."""
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/serving-monitor'
STOP = threading.Event()
LOCK = threading.RLock()
ROWS = collections.deque(maxlen=30000)
INTERVALS = collections.deque(maxlen=360)
CONTEXT = {}
STATUS = {}
CHILDREN = []


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(value)
    tmp.replace(path)


def event(kind, **values):
    row = {'time': time.time(), 'kind': kind, **values}
    name = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H')
    with LOCK:
        with (OUT / (name + '.jsonl')).open('a') as f:
            f.write(json.dumps(row, separators=(',', ':')) + '\n')
    return row


def quantile(values, q):
    values = sorted(values)
    if not values:
        return None
    n = (len(values)-1)*q
    i = int(n)
    return values[i] + (values[min(i+1,len(values)-1)]-values[i])*(n-i)


def metrics():
    with urllib.request.urlopen('http://127.0.0.1:30001/metrics', timeout=4) as f:
        content = f.read().decode()
    values = collections.defaultdict(float)
    for line in content.splitlines():
        m = re.match(r'([^ #{}]+)(?:\{(.*?)\})?\s+([-+0-9.eE]+)$', line)
        if not m:
            continue
        name, labels, value = m.groups()
        bucket = re.search(r'(?:^|,)le="([^"]+)"', labels or '')
        values[(name, bucket.group(1) if bucket else '')] += float(value)
    return values


def histogram_delta(before, after, prefix):
    n = after[(prefix+'_count','')] - before[(prefix+'_count','')]
    if n <= 0:
        return None
    total = after[(prefix+'_sum','')] - before[(prefix+'_sum','')]
    if total < 0:
        return None
    result = {'n': n, 'mean_ms': 1000*total/n}
    buckets = sorted((float(le), value-before[(name,le)]) for (name,le),value in after.items() if name == prefix+'_bucket')
    for q in (.5,.95):
        lower = 0
        for upper,count in buckets:
            if count >= n*q:
                result[f'p{int(q*100)}_bounds_ms'] = [1000*lower, 1000*upper if math.isfinite(upper) else None]
                break
            lower = upper
    return result


def accounting():
    while not STOP.is_set():
        try:
            proc = subprocess.Popen(['docker','logs','--follow','--since','2s','qwen38-kv-paging-r0'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            CHILDREN.append(proc)
            for line in proc.stdout:
                if STOP.is_set():
                    break
                if 'GB10_KV_ACCOUNTING ' not in line:
                    continue
                raw = json.loads(line.split('GB10_KV_ACCOUNTING ',1)[1])
                computed = [r['computed'] for r in raw['requests'].values() if 'computed' in r]
                with LOCK:
                    CONTEXT.clear()
                    CONTEXT.update(time=raw['time'], requests=len(computed), mean_tokens=statistics.mean(computed) if computed else 0,
                                   p95_tokens=quantile(computed,.95), total_tokens=sum(computed), pool_used=raw['pool_used'])
                    event('context', **CONTEXT)
            proc.terminate()
        except Exception as e:
            STATUS['accounting_error'] = str(e)
        STOP.wait(5)


def bpf_script(pid):
    root = f'/proc/{pid}/root/opt/gb10'
    return f'''
uprobe:{root}/libple_hash_gpu.so:gb10_ple_hash_gpu* /pid=={pid}/ {{ @reqs=arg3; @tokens=arg1; }}
uprobe:{root}/libple_batch_reader.so:gb10_ple_reader_gather /pid=={pid}/ {{ @sync=1; }}
uretprobe:{root}/libple_batch_reader.so:gb10_ple_reader_gather /pid=={pid}/ {{ @sync=0; }}
uprobe:{root}/libple_batch_reader.so:gb10_ple_reader_submit* /pid=={pid}/ {{ @read=nsecs; @rows=arg2; @h=arg0; @hits=*(uint64*)uptr(arg0+144); @miss=*(uint64*)uptr(arg0+120); @deferred=1-@sync; }}
uretprobe:{root}/libple_batch_reader.so:gb10_ple_reader_complete /pid=={pid}/ {{ @duration=nsecs-@read; @code=retval; @hits=*(uint64*)uptr(@h+144)-@hits; @miss=*(uint64*)uptr(@h+120)-@miss; }}
uprobe:{root}/libmapped_wait.so:gb10_release /pid=={pid}/ {{
 if (@lastseq==arg1 && @lastaddr==arg0+64) {{
   @done=arg0;
   printf("S %llu %llu %llu %llu %llu %llu %llu %llu %lld %llu %llu\\n", @start,arg1,@reqs,@tokens,@rows,@duration,@hits,@miss,@code,@deferred,nsecs);
 }} else {{
   @start=nsecs;
   if (@done && arg0==@done+64) {{
     printf("G %llu %llu %llu %llu %llu %llu\\n",*(uint64*)uptr(arg0+128),*(uint64*)uptr(arg0+112),*(uint64*)uptr(arg0+96),*(uint64*)uptr(arg0+72),*(uint64*)uptr(arg0+80),*(uint64*)uptr(arg0+64));
   }}
 }}
 @lastseq=arg1; @lastaddr=arg0;
}}
tracepoint:sched:sched_process_exit /pid=={pid} && tid=={pid}/ {{ exit(); }}
END {{ clear(@reqs); clear(@tokens); clear(@sync); clear(@read); clear(@rows); clear(@h); clear(@hits); clear(@miss); clear(@deferred); clear(@duration); clear(@code); clear(@lastseq); clear(@lastaddr); clear(@start); clear(@done); }}
'''


def rank_monitor(rank):
    prefix = [] if rank == 0 else ['ssh','-o','ConnectTimeout=5','192.168.100.11']
    expected = json.loads((ROOT/'experiments/ple-latency-campaign-20260916/deploy.fixed-sqpoll-r0.json').read_text())
    library_keys = ('ple_batch_library','ple_hash_library','ple_mapped_wait_library')
    while not STOP.is_set():
        proc = None
        try:
            inspected = json.loads(subprocess.check_output(prefix+['docker','inspect',f'qwen38-kv-paging-r{rank}'], text=True, timeout=10))[0]
            mounts = {m['Destination']:m['Source'] for m in inspected['Mounts']}
            names = ('libple_batch_reader.so','libple_hash_gpu.so','libmapped_wait.so')
            paths = [mounts['/opt/gb10/'+n] for n in names]
            hashes = subprocess.check_output(prefix+['sha256sum',*paths], text=True, timeout=10).splitlines()
            if any(line.split()[0] not in ({expected[key+'_sha256'], '522a29d866c7f9c93bb86a4e468774220736503ec0b229cb34887a813420d761'} if key == 'ple_batch_library' else {expected[key+'_sha256']}) for line,key in zip(hashes,library_keys)):
                raise RuntimeError('Unrecognized telemetry ABI: component monitoring disabled; API monitoring continues')
            top = subprocess.check_output(prefix+['docker','top',f'qwen38-kv-paging-r{rank}','-eo','pid,comm'],text=True,timeout=10)
            pid = int(next(l.split()[0] for l in top.splitlines() if 'VLLM::Worker' in l))
            tag = hashlib.sha256(json.dumps({'id':inspected['Id'],'hashes':hashes},sort_keys=True).encode()).hexdigest()[:16]
            script = OUT/f'rank{rank}.bt'
            script.write_text(bpf_script(pid))
            remote = f'/tmp/serving-monitor-r{rank}.bt'
            if rank:
                subprocess.run(['scp',str(script),'192.168.100.11:'+remote],check=True,stdout=subprocess.DEVNULL,timeout=10)
            command = prefix+['sudo','-n','bpftrace','-q',remote if rank else str(script)]
            err = (OUT/f'rank{rank}.stderr').open('a')
            proc = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
            CHILDREN.append(proc)
            STATUS[f'rank{rank}'] = {'pid':pid,'epoch':tag,'state':'attaching'}
            event('attachment',rank=rank,pid=pid,epoch=tag,hashes=hashes)
            pending = None
            for line in proc.stdout:
                if STOP.is_set():
                    break
                fields = line.split()
                if not fields or fields[0] not in ('S','G'):
                    continue
                values = list(map(int,fields[1:]))
                STATUS[f'rank{rank}']['last_sample'] = time.time()
                STATUS[f'rank{rank}']['state'] = 'collecting'
                if fields[0]=='G':
                    seq,start,end,wait,polls,error = values
                    if pending and pending['seq']==seq:
                        if 0 < start <= end and end-start < 10_000_000_000:
                            pending.update(gate_ms=(end-start)/1e6, wait_us=wait/1000, polls=polls, gpu_error=error)
                else:
                    start,seq,requests,tokens,rows,reader,hits,misses,error,deferred,done = values
                    now = time.time()
                    with LOCK:
                        context = dict(CONTEXT)
                        interval = dict(INTERVALS[-1]) if INTERVALS else {}
                    if pending:
                        if pending['seq']+1==seq:
                            pending['step_ms']=(start-pending['start_ns'])/1e6
                            pending['same_next_shape']=(pending['requests'],pending['tokens'],pending['rows'],pending['deferred'])==(requests,tokens,rows,deferred)
                        saved = event('step',**pending)
                        with LOCK:
                            ROWS.append(saved)
                    pending = {'time':now,'rank':rank,'epoch':tag,'seq':seq,'start_ns':start,'requests':requests,'tokens':tokens,'rows':rows,
                               'reader_ms':reader/1e6,'hits':hits,'misses':misses,'error':error,'deferred':deferred,
                               'context':context,'conditions':{k:interval.get(k) for k in ['time','prefill_tokens_s','acceptance','running']}}
            if proc:
                proc.terminate()
                proc.wait(timeout=5)
            err.close()
            STATUS[f'rank{rank}']['state']='disconnected'
        except Exception as e:
            STATUS[f'rank{rank}'] = {'state':'unavailable','error':str(e)}
            if proc and proc.poll() is None:
                proc.terminate()
        STOP.wait(10)


def render():
    now = time.time()
    with LOCK:
        intervals = [dict(r) for r in INTERVALS if now-r['time']<=300]
        rows = [dict(r) for r in ROWS if now-r['time']<=300]
        context = dict(CONTEXT)
    groups = collections.defaultdict(list)
    for row in rows:
        ctx = row['context']
        cond = row['conditions']
        if not row['deferred'] or not row.get('same_next_shape') or 'step_ms' not in row:
            continue
        if row['time']-ctx.get('time',0)>5 or row['time']-(cond.get('time') or 0)>10:
            continue
        # Accounting contains all resident requests; mismatches must not masquerade as exact context data.
        if ctx.get('requests') != row['requests'] or cond.get('running') != row['requests']:
            continue
        context_bin = int(ctx['mean_tokens']//4096)
        prefill = 'prefill' if (cond.get('prefill_tokens_s') or 0)>0 else 'decode'
        acceptance = cond.get('acceptance')
        if acceptance is None:
            continue
        bucket = min(9,int(acceptance*10))
        key = (row['epoch'],row['rank'],row['requests'],row['rows'],context_bin,prefill,bucket)
        groups[key].append(row)
    report_groups = []
    for key,values in sorted(groups.items(),key=lambda item:-len(item[1])):
        epoch,rank,requests,rows_count,ctx,prefill,acceptance = key
        blocks = len({int(r['time']//10) for r in values})
        item={'epoch':epoch,'rank':rank,'requests':requests,'rows':rows_count,'context_bin_tokens':[ctx*4096,(ctx+1)*4096],
              'mode':prefill,'acceptance_bin':[acceptance/10,(acceptance+1)/10],'n':len(values),'ten_second_blocks':blocks,
              'evidence':'descriptive' if len(values)>=100 and blocks>=5 else 'sparse'}
        for metric in ('step_ms','gate_ms','wait_us','reader_ms'):
            v=[r[metric] for r in values if metric in r]
            item[metric]={'n':len(v),'p50':quantile(v,.5),'p95':quantile(v,.95)}
        report_groups.append(item)
    current = intervals[-1] if intervals else {}
    result={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'window_seconds':300,'status':STATUS.copy(),
            'context':context,'current':current,'sampled_steps':len(rows),'eligible_steps':sum(g['n'] for g in report_groups),
            'native_errors':sum(r['error']!=0 for r in rows),'gpu_errors':sum(r.get('gpu_error',0)!=0 for r in rows),'groups':report_groups,
            'notes':['Rolling descriptive monitoring, not a causal ranking of implementations.','CPU publish-to-next-publish interval is a step proxy, not CUDA kernel duration.',
                     'GPU hash-to-ready gate includes intervening graph work; it is not L2 residency.','GPU wait is wait-loop body time.',
                     'Context and speculative acceptance are recent aggregate observations, not exact per-step attribution.',
                     'Completed-request time/output-token spans the request lifetime, not just this scrape interval.', 'All-request burst means must not be compared across different workload groups.','Uprobes add overhead; no overhead-free claim. ABI-unknown implementations retain API monitoring only.']}
    atomic(OUT/'latest.json',json.dumps(result,indent=2)+'\n')
    lines=['# Live serving monitor',result['updated_utc'],'',f"Rolling 5 minutes: {len(rows)} rank-step samples; {result['eligible_steps']} in workload groups; native errors {result['native_errors']}, GPU errors {result['gpu_errors']}.",'']
    if current:
        lines += [f"Running {current.get('running')} | waiting {current.get('waiting')} | generated tokens/s {current.get('tokens_s',0):.1f} | prefill tokens/s {current.get('prefill_tokens_s',0):.1f}",
                  f"Recent context mean {context.get('mean_tokens',0):.0f} tokens; speculative acceptance {current.get('acceptance')}",
                  f"Existing time/output-token metric (requests completed in last interval): {current.get('output_token')}",
                  f"Existing output-burst interval (all workloads): {current.get('burst')}",'']
    lines += ['| Rank | Requests | Mean context bin | Recent work | Acceptance bin | Samples / 10s blocks | Step p50 / p95 ms | PLE gate p50 ms | GPU wait p95 µs | Evidence |',
              '|---|---:|---|---|---|---|---|---:|---:|---|']
    def fmt(v):
        return 'n/a' if v is None else f'{v:.3f}'
    for g in report_groups[:20]:
        lines.append(f"| {g['rank']} | {g['requests']} | {g['context_bin_tokens']} | {g['mode']} | {g['acceptance_bin']} | {g['n']} / {g['ten_second_blocks']} | {fmt(g['step_ms']['p50'])} / {fmt(g['step_ms']['p95'])} | {fmt(g['gate_ms']['p50'])} | {fmt(g['wait_us']['p95'])} | {g['evidence']} |")
    lines += ['', 'No winner is inferred from aggregate throughput. Compare matching groups and repeat time blocks; sparse or absent groups are inconclusive.', '', 'Probe status: '+json.dumps(STATUS), '', *result['notes']]
    atomic(OUT/'latest.md','\n'.join(lines)+'\n')


def cleanup():
    STOP.set()
    commands = [
        ['sudo', '-n', 'pkill', '-INT', '-f', '^bpftrace -q ' + str(OUT/'rank0.bt') + '$'],
        ['ssh', '-o', 'ConnectTimeout=5', '192.168.100.11',
         "sudo -n pkill -INT -f '^bpftrace -q /tmp/serving-monitor-r1.bt$'"],
    ]
    for command in commands:
        try:
            subprocess.run(command, timeout=8, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            pass
    for child in CHILDREN:
        if child.poll() is None:
            try:
                child.terminate()
            except ProcessLookupError:
                pass


def run():
    OUT.mkdir(exist_ok=True,parents=True)
    for target,args in [(accounting,()),(rank_monitor,(0,)),(rank_monitor,(1,))]:
        threading.Thread(target=target,args=args,daemon=True).start()
    before = None
    previous = time.monotonic()
    while not STOP.is_set():
        try:
            after = metrics()
            now = time.monotonic()
            if before is not None:
                dt = now-previous
                def delta(name):
                    return after[(name,'')]-before[(name,'')]
                if delta('vllm:generation_tokens_total')<0:
                    event('counter_reset')
                else:
                    drafted=delta('vllm:spec_decode_num_draft_tokens_total')
                    row=event('interval',seconds=dt,running=after[('vllm:num_requests_running','')],waiting=after[('vllm:num_requests_waiting','')],
                              tokens_s=delta('vllm:generation_tokens_total')/dt,prefill_tokens_s=delta('vllm:prompt_tokens_total')/dt,
                              acceptance=delta('vllm:spec_decode_num_accepted_tokens_total')/drafted if drafted>0 else None,
                              preemptions=delta('vllm:num_preemptions_total'),cache_fraction=after[('vllm:kv_cache_usage_perc','')],
                              output_token=histogram_delta(before,after,'vllm:request_time_per_output_token_seconds'),burst=histogram_delta(before,after,'vllm:inter_token_latency_seconds'),ttft=histogram_delta(before,after,'vllm:time_to_first_token_seconds'))
                    with LOCK:
                        INTERVALS.append(row)
            before,previous=after,now
            STATUS['metrics']='healthy'
        except Exception as e:
            STATUS['metrics']=str(e)
            before=None
        render()
        # Retain 24 hours of raw measurements; never touch serving files.
        for path in OUT.glob('20*.jsonl'):
            if time.time()-path.stat().st_mtime>86400:
                path.unlink()
        STOP.wait(5)
    for child in CHILDREN:
        if child.poll() is None:
            child.terminate()


if __name__=='__main__':
    signal.signal(signal.SIGTERM,lambda *_:STOP.set())
    signal.signal(signal.SIGINT,lambda *_:STOP.set())
    try:
        run()
    finally:
        cleanup()
