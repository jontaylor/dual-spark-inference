"""Change only image identity, preserving the current deployment's other settings."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('arm', choices=['stock', 'candidate'])
parser.add_argument('--config', type=Path, required=True)
parser.add_argument('--identities', type=Path, required=True)
args = parser.parse_args()
identities = json.loads(args.identities.read_text())
cfg = json.loads(args.config.read_text())
if cfg['image'] not in {v['image'] for v in identities.values()}:
    raise SystemExit('Another task changed the image; re-check before switching.')
before = {'image': cfg['image'], 'image_ids': cfg['image_ids']}
cfg.update(identities[args.arm])
tmp = args.config.with_suffix('.qsa-tmp.json')
tmp.write_text(json.dumps(cfg, indent=2) + '\n')
tmp.replace(args.config)
print(json.dumps({'before': before, 'after': identities[args.arm]}))
