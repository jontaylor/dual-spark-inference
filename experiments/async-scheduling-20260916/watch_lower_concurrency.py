"""Bounded observer: capture the unchanged workload once it reaches 8–14 requests.

No admission changes or requests. Exits after one capture or 30 minutes.
Checks worker process identity again before attaching measurement tools.
"""
import argparse
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--duration', type=int, default=1800)
parser.add_argument('--min-running', type=int, default=8)
parser.add_argument('--max-running', type=int, default=14)
parser.add_argument('--cuda-waits', action='store_true')
parser.add_argument('--output', type=Path, default=ROOT/'results/async-scheduling-20260916/lower-concurrency')
options = parser.parse_args()
if not 1 <= options.duration <= 7200:
    parser.error('duration must be between 1 and 7200 seconds')
if not 1 <= options.min_running <= options.max_running <= 32:
    parser.error('running range must be within 1..32')
OUT = options.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
REMOTE = 'jon@192.168.100.11'
PIDS = ()


def call(rank, args, **kwargs):
    return subprocess.run(args if rank == 0 else ['ssh', REMOTE, shlex.join(args)],
                          check=True, **kwargs)


def identity(rank):
    code = ('import json,time;from pathlib import Path;'
            f'p=Path("/proc/{PIDS[rank]}/stat").read_text().rsplit(")",1)[1].split();'
            'print(json.dumps({"start_ticks":p[19],'
            '"wall_minus_monotonic_ns":time.time_ns()-time.monotonic_ns()}))')
    return json.loads(call(rank, ['python3', '-c', code], capture_output=True,
                           text=True, timeout=15).stdout)


def worker_pid(rank):
    output = call(rank, ['docker', 'top', f'qwen38-kv-paging-r{rank}', '-eo', 'pid,comm'],
                  capture_output=True, text=True, timeout=15).stdout
    workers = [int(line.split()[0]) for line in output.splitlines()
               if 'VLLM::Worker' in line]
    if len(workers) != 1:
        raise RuntimeError(f'Expected one rank-{rank} GPU worker, found {workers}')
    return workers[0]


def capture(rank, initial):
    current = identity(rank)
    if current['start_ticks'] != initial['start_ticks']:
        raise RuntimeError(f'Rank {rank} worker changed; do not attach')
    base = str(OUT/f'gpu-metrics-r{rank}') if rank == 0 else '/tmp/async-low-gpu-metrics-r1'
    probe = (ROOT/'experiments/async-scheduling-20260916/host-gap.bt').read_text()
    probe = probe.replace('456247', str(PIDS[rank]))
    if options.cuda_waits:
        extra = (ROOT/'experiments/async-scheduling-20260916/cuda-waits.bt').read_text()
        extra = extra.replace('interval:s:60 { exit(); }', '').replace('END { clear(@start); }', '')
        probe = probe.replace('clear(@end);', 'clear(@end); clear(@start);')
        probe += '\n' + extra.replace('456247', str(PIDS[rank]))
    local_probe = OUT/f'host-gap-r{rank}.bt'
    local_probe.write_text(probe)
    probe_path = str(local_probe)
    if rank:
        probe_path = '/tmp/async-low-host-gap.bt'
        subprocess.run(['scp', '-q', str(local_probe), f'{REMOTE}:{probe_path}'], check=True)
    probe_file = (OUT/f'host-gap-metrics-r{rank}.txt').open('w')
    log = (OUT/f'gpu-metrics-r{rank}.log').open('w')
    cmd = ['sudo', '-n', 'bpftrace', '-q', probe_path]
    process = subprocess.Popen(cmd if rank == 0 else ['ssh', REMOTE, shlex.join(cmd)],
                               stdout=probe_file, stderr=subprocess.STDOUT)
    try:
        call(rank, ['sudo', '-n', 'nsys', 'profile', '--trace=none', '--sample=none',
                    '--cpuctxsw=none', '--gpu-metrics-devices=0',
                    '--gpu-metrics-frequency=10000', '--duration=30',
                    '--force-overwrite=true', '--output='+base],
             stdout=log, stderr=subprocess.STDOUT, timeout=120)
        # Probe contains its own 60-second deadline, including on the remote host.
        process.wait(timeout=75)
        if process.returncode:
            raise RuntimeError(f'Rank {rank} probe failed: {process.returncode}')
    finally:
        if process.poll() is None:
            process.terminate()
        probe_file.close()
        log.close()
    final = identity(rank)
    if final['start_ticks'] != initial['start_ticks']:
        raise RuntimeError(f'Rank {rank} worker changed during capture')
    (OUT/('clock.json' if rank == 0 else 'clock-r1.json')).write_text(json.dumps(final))
    if rank:
        subprocess.run(['scp', '-q', f'{REMOTE}:{base}.nsys-rep',
                        str(OUT/'gpu-metrics-r1.nsys-rep')], check=True)
    with (OUT/f'export-r{rank}.log').open('w') as export_log:
        subprocess.run(['nsys', 'export', '--type=sqlite', '--force-overwrite=true',
                        '--output='+str(OUT/f'gpu-metrics-r{rank}.sqlite'),
                        str(OUT/f'gpu-metrics-r{rank}.nsys-rep')],
                       check=True, stdout=export_log, stderr=subprocess.STDOUT, timeout=90)


PIDS = tuple(worker_pid(rank) for rank in (0, 1))
initial = [identity(rank) for rank in (0, 1)]
started = time.monotonic()
stable = 0
pool = None
futures = []
status = {'state': 'waiting', 'start_wall': time.time(), 'duration_seconds': options.duration,
          'min_running': options.min_running, 'max_running': options.max_running}
try:
    with (OUT/'requests.jsonl').open('a') as observations:
        while time.monotonic()-started < options.duration:
            try:
                with urllib.request.urlopen('http://127.0.0.1:30001/metrics', timeout=5) as response:
                    raw = response.read().decode()
                running = sum(float(line.split()[-1]) for line in raw.splitlines()
                              if line.startswith('vllm:num_requests_running{'))
                waiting = sum(float(line.split()[-1]) for line in raw.splitlines()
                              if line.startswith('vllm:num_requests_waiting{'))
                observations.write(json.dumps({'time': time.time(), 'running': running,
                                               'waiting': waiting})+'\n')
                observations.flush()
                status.update(updated_wall=time.time(), running=running)
                if not futures:
                    stable = stable+1 if options.min_running <= running <= options.max_running else 0
                    if stable >= 2:
                        status['state'] = 'capturing'
                        pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
                        futures = [pool.submit(capture, rank, initial[rank]) for rank in (0, 1)]
                elif all(f.done() for f in futures):
                    for future in futures:
                        future.result()
                    subprocess.run([str(ROOT/'.venv/bin/python'),
                                    str(Path(__file__).with_name('analyze.py')), str(OUT)],
                                   check=True, stdout=subprocess.DEVNULL)
                    status['state'] = 'complete'
                    break
            except (OSError, ValueError) as exc:
                status['last_observation_error'] = str(exc)
            (OUT/'status.json').write_text(json.dumps(status, indent=2))
            time.sleep(2 if futures else 15)
        else:
            status['state'] = 'expired'
except Exception as exc:
    status.update(state='failed', error=str(exc))
    raise
finally:
    if pool:
        pool.shutdown(wait=True)
    status['updated_wall'] = time.time()
    (OUT/'status.json').write_text(json.dumps(status, indent=2))
print(json.dumps(status))
