#!/usr/bin/env python3
"""Load the existing credential inside the head container without logging it."""
import os
import sys
from pathlib import Path

if "--headless" not in sys.argv:
    key = Path('/run/secrets/inference-api-key').read_text().strip()
    if not key:
        raise RuntimeError('Existing API key is empty')
    os.environ['VLLM_API_KEY'] = key
args = ['vllm', 'serve', *sys.argv[1:]]
if os.environ.get('QWEN_NSYS_CAPTURE') == '1':
    import time
    nsys = '/opt/nsight/target-linux-sbsa-armv8/nsys'
    args = [nsys, 'profile', '--trace=cuda,nvtx,osrt', '--sample=none',
            '--cpuctxsw=none', '--cuda-graph-trace=node',
            '--capture-range=cudaProfilerApi', '--capture-range-end=repeat:20',
            '--kill=none', '--output=/profiles/timeline-' + str(time.time_ns()),
            *args]
    os.execv(nsys, args)
os.execvp('vllm', args)
