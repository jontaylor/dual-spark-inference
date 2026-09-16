"""Verify both live containers against a selected campaign config pair."""
import argparse
import json
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--config-prefix', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
base = Path(__file__).resolve().parent
package = '/usr/local/lib/python3.12/dist-packages/vllm/'
results = {}
errors = []
for rank in (0, 1):
    config = json.loads((base / f'{args.config_prefix}-r{rank}.json').read_text())
    prefix = [] if rank == 0 else ['ssh', '192.168.100.11']
    container = f'qwen38-kv-paging-r{rank}'
    inspected = json.loads(subprocess.check_output(prefix + ['docker', 'inspect', container], text=True))[0]
    actual = {m['Destination']: m['Source'] for m in inspected['Mounts']}
    expected = {package + relative: source for relative, source in config['runtime_overrides'].items()}
    for key, filename in [('ple_batch_library', 'libple_batch_reader.so'), ('ple_hash_library', 'libple_hash_gpu.so'), ('ple_mapped_wait_library', 'libmapped_wait.so')]:
        if key in config:
            expected['/opt/gb10/' + filename] = config[key]
    mismatches = {destination: {'expected': source, 'actual': actual.get(destination)} for destination, source in expected.items() if actual.get(destination) != source}
    environment = [entry for entry in inspected['Config']['Env'] if entry.startswith(('GB10_PLE_', 'QWEN_NSYS_'))]
    values = dict(entry.split('=', 1) for entry in environment)
    options = config['optimizations']
    expected_env = {'GB10_PLE_IN_PROCESS': '1', 'GB10_PLE_READ_CONTROL':'/opt/gb10/ple-read-policy.bin'}
    for option, key in [('ple_overlap', 'GB10_PLE_OVERLAP'), ('ple_gpu_hash', 'GB10_PLE_GPU_HASH'), ('ple_mapped_transport', 'GB10_PLE_MAPPED_TRANSPORT')]:
        if options.get(option):
            expected_env[key] = '1'
    if 'ple_submission_policy' in options:
        expected_env['GB10_PLE_SUBMISSION_POLICY'] = options['ple_submission_policy']
    if options.get('ple_async_ab_steps'):
        expected_env['GB10_PLE_ASYNC_AB_STEPS'] = str(options['ple_async_ab_steps'])
    elif values.get('GB10_PLE_ASYNC_AB_STEPS') not in (None, '0'):
        errors.append(f'rank{rank}: experimental switching enabled unexpectedly')
    for key, value in expected_env.items():
        if values.get(key) != value:
            errors.append(f'rank{rank}: {key} does not match intended configuration')
    argv = inspected['Config']['Cmd']
    parameters = {}
    for key, flag in [('max_num_seqs', '--max-num-seqs'), ('max_num_batched_tokens', '--max-num-batched-tokens'), ('long_prefill_token_threshold', '--long-prefill-token-threshold'), ('tensor_parallel_size', '--tensor-parallel-size')]:
        if key in config:
            actual_value = argv[argv.index(flag) + 1] if flag in argv else None
            parameters[key] = actual_value
            if actual_value != str(config[key]):
                errors.append(f'rank{rank}: {flag} does not match intended configuration')
    if config.get('mtp_tokens'):
        speculative = json.loads(argv[argv.index('--speculative-config') + 1]) if '--speculative-config' in argv else {}
        parameters['mtp_tokens'] = speculative.get('num_speculative_tokens')
        if parameters['mtp_tokens'] != config['mtp_tokens']:
            errors.append(f'rank{rank}: MTP token count does not match configuration')
    processes = subprocess.check_output(prefix + ['docker', 'top', container, '-eo', 'pid,comm'], text=True)
    results[rank] = {'mounts': {key: actual.get(key) for key in expected}, 'mismatches': mismatches, 'env': environment, 'parameters': parameters, 'processes': processes}
    if mismatches:
        errors.append(f'rank{rank}: {len(mismatches)} mount mismatches')
    if 'GB10_PLE_IN_PROCESS=1' not in environment:
        errors.append(f'rank{rank}: in-process mode missing')
args.output.write_text(json.dumps(results, indent=2) + '\n')
if errors:
    raise SystemExit('; '.join(errors))
print('PASS both ranks: runtime mounts, policy settings and serving parameters')
