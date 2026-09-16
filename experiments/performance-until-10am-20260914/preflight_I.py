"""Read-only MTP5 preflight; does not stop, deploy, or resume anything."""
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

p = Path(__file__).resolve().parent
root = p.parents[1]
expected = 'ee5e88d3f660cbe261dd8ec4b6ee78bcc17abb6a50d4aabfd59d92347fc7d35d'
assert hashlib.sha256((root / 'deploy_config.json').read_bytes()).hexdigest() == expected
paired = json.loads((p / 'readback-paired-H/summary.json').read_text())
assert paired['status'] == 'complete' and paired['restored_full_verification']
assert len(paired['phases']) == 8 and all(r['probe_exact'] for r in paired['phases'])
rows = []
for rank in (0, 1):
    candidate = json.loads((p / f'candidate-I-mtp5-r{rank}.json').read_text())
    baseline = json.loads((p / f'candidate-H-r{rank}.json').read_text())
    changed = {k for k in candidate if candidate[k] != baseline.get(k)}
    assert changed == {'mtp_tokens', 'cudagraph_capture_sizes'}
    assert candidate['mtp_tokens'] == 5
    assert max(candidate['cudagraph_capture_sizes']) >= candidate['max_num_seqs'] * 6
    for name, source in candidate['runtime_overrides'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == candidate['runtime_override_sha256'][name]
    code = """import hashlib,json,sys
from pathlib import Path
d=json.load(sys.stdin)
for name,source in d['runtime_overrides'].items():
 assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==d['runtime_override_sha256'][name],name
print(json.dumps({'source_hashes_verified':len(d['runtime_overrides'])}))
"""
    cmd = ['python3', '-c', code]
    if rank:
        cmd = ['ssh', 'jon@192.168.100.11', shlex.join(cmd)]
    source_result = json.loads(subprocess.check_output(cmd, input=json.dumps(candidate), text=True, timeout=30))
    cmd = ['docker', 'exec', f'qwen38-kv-paging-r{rank}', 'cat', '/tmp/vllm-kv-readback-control.json']
    if rank:
        cmd = ['ssh', 'jon@192.168.100.11', shlex.join(cmd)]
    assert json.loads(subprocess.check_output(cmd, text=True, timeout=20))['gpu_readback'] is True
    rows.append({'rank': rank, **source_result, 'readback_enabled': True})
result = {'time': time.time(), 'passed': True, 'ranks': rows,
          'scope': 'Configuration and source readiness only. Does not establish model correctness or authorize restart before a clean workload boundary.',
          'remaining': ['verify controller hold identity and all batch owners exited',
                        'confirm backend idle', 'stop and finalize physical observers',
                        'save exact H rollback and final counters', 'deploy and run live I gates']}
(p / 'I-preflight.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
