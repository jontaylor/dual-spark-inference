"""Short API correctness/authentication checks; this is not a benchmark."""
import concurrent.futures
import json
from runtime_config import load_config, model_snapshot, resolve_path
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parent
cfg = load_config(root)
base = f"http://127.0.0.1:{cfg['port']}"
key = resolve_path(cfg['api_key_file'], root).read_text().strip()
headers = {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}

def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)

with urllib.request.urlopen(base + '/health', timeout=10) as response:
    assert response.status == 200
try:
    urllib.request.urlopen(base + '/v1/models', timeout=10)
    raise AssertionError('Unauthenticated request unexpectedly succeeded')
except urllib.error.HTTPError as exc:
    assert exc.code == 401, exc.code
models = [item['id'] for item in request('/v1/models')['data']]
assert set(cfg['served_names']).issubset(models)

def chat(model, prompt, **kwargs):
    return request('/v1/chat/completions', {
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 128,
        'temperature': 0,
        'chat_template_kwargs': {'enable_thinking': False},
        **kwargs,
    })

def arithmetic(i):
    a, b = 17 + i, 19 + i
    start = time.monotonic()
    model = cfg['served_names'][i % len(cfg['served_names'])]
    result = chat(model, f'What is {a} times {b}? Reply with just the number.')
    content = result['choices'][0]['message']['content'].strip()
    assert content == str(a * b), repr(content)
    return {'model': model, 'response': content,
            'seconds': round(time.monotonic() - start, 2), 'usage': result['usage']}

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    checks = list(pool.map(arithmetic, range(4)))

tool = {'type': 'function', 'function': {
    'name': 'get_weather', 'description': 'Get the current weather for a city.',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}},
                   'required': ['city']},
}}
result = chat(cfg['served_names'][0], 'Use the weather tool to get the current weather in London.',
              tools=[tool], tool_choice='auto')
calls = result['choices'][0]['message'].get('tool_calls')
assert calls and calls[0]['function']['name'] == 'get_weather', result
args = json.loads(calls[0]['function']['arguments'])
assert args['city'].lower() == 'london', args

payload = {'model': cfg['served_names'][0], 'stream': True,
           'messages': [{'role': 'user', 'content': 'Reply with exactly READY'}],
           'max_tokens': 16, 'temperature': 0,
           'chat_template_kwargs': {'enable_thinking': False}}
req = urllib.request.Request(base + '/v1/chat/completions',
                             data=json.dumps(payload).encode(), headers=headers)
parts, done = [], False
with urllib.request.urlopen(req, timeout=90) as response:
    for line in response:
        if not line.startswith(b'data: '):
            continue
        value = line[6:].strip()
        if value == b'[DONE]':
            done = True
            break
        event = json.loads(value)
        for choice in event.get('choices', []):
            parts.append(choice.get('delta', {}).get('content') or '')
assert done and ''.join(parts).strip() == 'READY', parts

report = {'health': 200, 'unauthenticated_models': 401, 'models': models,
          'concurrent_arithmetic_checks': checks,
          'tool_call': {'name': 'get_weather', 'arguments': args},
          'streaming_response': ''.join(parts),
          'note': 'Short correctness checks only; no full-context or throughput claim.'}
Path(sys.argv[1]).write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
