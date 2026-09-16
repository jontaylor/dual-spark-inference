"""Collect comparison records without starting requests or changing serving."""

import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/ple-backend-comparison-20260916"
LOAD = "/home/jon/artifact-agent-orchestration/.artifact-agent/ple-comparison/20260916-fixed-decode"


def remote_json(host, script, args=()):
    command = shlex.join(["python3", "-c", script, *map(str, args)])
    return json.loads(subprocess.check_output(["ssh", host, command], text=True, timeout=30))


def main():
    script = '''
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);result={}
for name in ('cycles.jsonl','requests.jsonl','status.json','config.json','comparison-plan.json'):
 f=p/name
 if not f.exists():continue
 if name.endswith('.jsonl'):
  result[name]=[json.loads(line) for line in f.read_text().splitlines() if line]
 else:result[name]=json.loads(f.read_text())
print(json.dumps(result))
'''
    client = remote_json("jon@192.168.0.167", script, (LOAD,))
    (OUT / "client.json").write_text(json.dumps(client, indent=2) + "\n")
    command = shlex.join(["sudo", "-n", "tar", "-C", str(OUT / "live/rank1"), "-cf", "-", "."])
    result = subprocess.run(["ssh", "jon@192.168.100.11", command],
                            capture_output=True, check=True, timeout=30)
    destination = OUT / "live/rank1"
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "--no-same-owner", "-C", str(destination), "-xf", "-"],
                   input=result.stdout, check=True, timeout=30)
    print(json.dumps({"client_state": client.get("status.json"),
                      "cycles": len(client.get("cycles.jsonl", []))}))


if __name__ == "__main__":
    main()
