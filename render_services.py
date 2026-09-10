#!/usr/bin/env python3
"""Render systemd units for review; this script does not install or start them."""
import argparse
import getpass
import grp
import os
from pathlib import Path
from runtime_config import load_config

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--user', default=getpass.getuser())
parser.add_argument('--group', default=grp.getgrgid(os.getgid()).gr_name)
parser.add_argument('--output', type=Path, default=root / 'rendered-services')
args = parser.parse_args()
cfg = load_config(root)
# SSH remote command and systemd token fields must remain single safe tokens.
import re
for value in (args.user, args.group, str(root), cfg['worker_ip']):
    if not re.fullmatch(r'[A-Za-z0-9_./:@-]+', value):
        raise SystemExit('Use simple paths and a hostname/IP without whitespace or systemd substitutions')
args.output.mkdir(parents=True, exist_ok=True)
common = f'''[Unit]
Description=Qwen NVIDIA NVFP4 / BF16 KV with NVMe paging on dual GB10
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=simple
User={args.user}
Group={args.group}
WorkingDirectory={root}
Restart=no
TimeoutStartSec=3600
TimeoutStopSec=90
'''
for rank, name in [(0, 'qwen38-next-qwen-fp8'), (1, 'qwen38-next-qwen-fp8-worker')]:
    text = common
    if rank == 0:
        text += f'ExecStartPre=/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15 {cfg["worker_ip"]} sudo -n systemctl start qwen38-next-qwen-fp8-worker.service\n'
    text += f'ExecStart={root}/.venv/bin/python {root}/supervise_rank.py {rank}\n'
    if rank == 0:
        text += f'ExecStopPost=-/usr/bin/ssh -o BatchMode=yes -o ConnectTimeout=15 {cfg["worker_ip"]} sudo -n systemctl stop qwen38-next-qwen-fp8-worker.service\n'
    text += '\n[Install]\nWantedBy=multi-user.target\n'
    (args.output / (name + '.service')).write_text(text)
print(f'Rendered units in {args.output}')
