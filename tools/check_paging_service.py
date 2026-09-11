"""Acceptance checks for the configured live service; uses the existing local key."""
import asyncio
import json
import time
from pathlib import Path

import aiohttp
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / 'deploy_config.json').read_text())
OUT = ROOT / 'results' / 'paging-handover'
OUT.mkdir(parents=True, exist_ok=True)
BASE = f"http://127.0.0.1:{CFG['port']}"
HEADERS = {'Authorization': 'Bearer ' + Path(CFG['api_key_file']).read_text().strip()}
MODEL = CFG['served_names'][0]

async def main():
    samples = []
    finished = asyncio.Event()
    async with aiohttp.ClientSession(headers=HEADERS, timeout=aiohttp.ClientTimeout(total=2400)) as session:
        async def monitor():
            while not finished.is_set():
                async with session.get(BASE + '/metrics') as response:
                    text = await response.text()
                    selected = [line for line in text.splitlines() if not line.startswith('#') and any(x in line for x in ('num_requests_running{', 'num_requests_waiting{', 'kv_cache_usage_perc{', 'num_preemptions_total{'))]
                    samples.append({'time': time.time(), 'metrics': selected})
                await asyncio.sleep(1)

        async def request(label, prompt, generated):
            started = time.monotonic()
            async with session.post(BASE + '/v1/completions', json={
                'model': MODEL, 'prompt': prompt, 'temperature': 0,
                'max_tokens': generated, 'min_tokens': generated,
            }) as response:
                payload = await response.json()
                assert response.status == 200, (response.status, payload)
                assert payload['usage']['completion_tokens'] == generated, payload['usage']
                result = {'label': label, 'elapsed': time.monotonic() - started, 'usage': payload['usage'], 'text': payload['choices'][0]['text']}
                (OUT / f'{label}.json').write_text(json.dumps(result, indent=2))
                print(label, result['usage'], round(result['elapsed'], 2), flush=True)
                return result

        for auth in (None, 'Bearer invalid-test-key'):
            async with aiohttp.ClientSession() as unauth:
                async with unauth.post(BASE + '/v1/completions', headers={} if auth is None else {'Authorization': auth}, json={'model': MODEL, 'prompt': 'test', 'max_tokens': 1}) as response:
                    assert response.status == 401, response.status
        task = asyncio.create_task(monitor())
        try:
            small = await asyncio.gather(*(request(f'c{CFG["max_num_seqs"]}-{i:02}', f'Request {i}. Explain the benefits and limitations of database indexes in detail.', 256) for i in range(CFG['max_num_seqs'])))
            peak = max(float(line.rsplit(' ', 1)[1]) for sample in samples for line in sample['metrics'] if 'num_requests_running{' in line)
            assert peak == CFG['max_num_seqs'], peak
            print('Configured peak active', peak, flush=True)
            snapshot = Path(CFG['paths']['hf_cache']) / ('models--' + CFG['model_id'].replace('/', '--')) / 'snapshots' / CFG['revision']
            tok = Tokenizer.from_file(str(snapshot / 'tokenizer.json'))
            fixture = json.loads(Path('/home/jon/gb10-optimisation/20260910-placement-live/fixture-12000.json').read_text())['source']
            ids = tok.encode(fixture, add_special_tokens=False).ids
            prefix = tok.encode('Review this code and describe its design.\n', add_special_tokens=False).ids
            suffix = tok.encode('\nNow give the review:\n', add_special_tokens=False).ids
            length = 262016
            tokens = prefix + (ids * ((length // len(ids)) + 1))[:length - len(prefix) - len(suffix)] + suffix
            assert len(tokens) == length
            long = await request('context-262016', tokens, 128)
            assert long['usage']['prompt_tokens'] == length
            async with session.get(BASE + '/health') as response:
                assert response.status == 200
            (OUT / 'summary.json').write_text(json.dumps({'peak_active': peak, 'small_completed': len(small), 'long': long, 'authentication': 'missing and wrong key rejected'}, indent=2))
        finally:
            finished.set()
            await task
            (OUT / 'metrics.json').write_text(json.dumps(samples))

asyncio.run(main())
