"""Replay the archived payload unchanged except request ID and optional seed."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
MODE = sys.argv[1]
OUT = ROOT / MODE
OUT.mkdir(exist_ok=False)
PAYLOAD = json.loads((ROOT / 'request.json').read_text())
if MODE.endswith('seed'):
    PAYLOAD['seed'] = 917352
KEY = Path('/home/jon/.config/qwen38/api-key').read_text().strip()
BASE = 'http://127.0.0.1:30001'
(OUT / 'request.json').write_text(json.dumps(PAYLOAD, indent=2))


def control(phase, enabled):
    data = json.dumps({'enabled': enabled, 'phase': MODE + '-' + phase,
                       'max_generated': 4, 'max_batches': 160})
    for host, name in [(None, 'qwen38-kv-paging-r0'),
                       ('192.168.100.11', 'qwen38-kv-paging-r1')]:
        cmd = ['docker', 'exec', '-i', name, 'python3', '-c',
               "from pathlib import Path; import sys; Path('/tmp/vllm-row-trace-control.json').write_text(sys.stdin.read())"]
        if host:
            cmd = ['ssh', '-o', 'BatchMode=yes', host, shlex.join(cmd)]
        subprocess.run(cmd, input=data, text=True, check=True, timeout=15)


def request(name):
    payload = dict(PAYLOAD, request_id='rowtrace-' + MODE + '-' + name)
    if MODE.startswith('cold'):
        import uuid
        payload['cache_salt'] = str(uuid.uuid4())
    req = urllib.request.Request(BASE + '/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={
            'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
    start = time.time()
    chunks = []
    error = None
    try:
        with urllib.request.urlopen(req, timeout=180) as response, (OUT / (name+'.sse')).open('wb') as f:
            for line in response:
                f.write(line)
                if line.startswith(b'data: ') and line.strip() != b'data: [DONE]':
                    chunks.append(json.loads(line[6:]))
    except Exception as exc:
        error = repr(exc)
    tokens = []
    logprobs = []
    reasoning = content = ''
    tools = {}
    usage = None
    prompt_ids = None
    for chunk in chunks:
        usage = chunk.get('usage') or usage
        prompt_ids = chunk.get('prompt_token_ids') or prompt_ids
        for choice in chunk.get('choices', []):
            tokens.extend(choice.get('token_ids') or [])
            logprobs.extend((choice.get('logprobs') or {}).get('content') or [])
            d = choice.get('delta') or {}
            reasoning += d.get('reasoning') or d.get('reasoning_content') or ''
            content += d.get('content') or ''
            for t in d.get('tool_calls') or []:
                f = tools.setdefault(t['index'], {'name': '', 'arguments': ''})
                f['name'] += t.get('function', {}).get('name', '')
                f['arguments'] += t.get('function', {}).get('arguments', '')
    canonical = {'reasoning': reasoning, 'content': content, 'tools': list(tools.values())}
    row = {'name': name, 'request_id': payload['request_id'], 'cache_salt': payload.get('cache_salt'), 'started': start,
           'seconds': time.time()-start, 'error': error, 'usage': usage,
           'prompt_token_ids': prompt_ids, 'token_ids': tokens, 'logprobs': logprobs,
           'canonical': canonical,
           'hash': hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest(),
           'chunks': chunks}
    (OUT / (name+'.json')).write_text(json.dumps(row, indent=2))
    print(name, len(tokens), row['hash'][:12], error, flush=True)
    return row


if MODE == 'serialized':
    import threading
    gate = threading.Lock()
    original_request = request
    def request(name):
        with gate:
            return original_request(name)

rows = []
try:
    for phase in ['serial-before', 'parallel', 'serial-after']:
        with urllib.request.urlopen(BASE + '/metrics', timeout=5) as r:
            metrics = r.read().decode()
        (OUT / (phase+'-metrics.txt')).write_text(metrics)
        busy = [l for l in metrics.splitlines()
                if l.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))
                and float(l.rsplit(' ', 1)[1]) != 0]
        if busy:
            raise RuntimeError('Other work is active: ' + repr(busy))
        control(phase, MODE.startswith('captured'))
        if phase == 'parallel':
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                rows.extend(pool.map(request, [phase+'-'+str(i) for i in range(4)]))
        else:
            rows.extend(request(phase+'-'+str(i)) for i in range(3))
        (OUT / 'results.json').write_text(json.dumps(rows, indent=2))
finally:
    control('finished', False)
