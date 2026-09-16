"""Run the agreed held/drained repeating-load plan, never restart serving."""

import json
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

from collect import LOAD, remote_json
from set_backend import idle

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT = ROOT / "results/ple-backend-comparison-20260916"
PYTHON = str(ROOT / ".venv/bin/python")
HOST = "jon@192.168.0.167"


def status():
    return remote_json(HOST, "import json,pathlib,sys; print((pathlib.Path(sys.argv[1])/'status.json').read_text())", (LOAD,))


def admit(control):
    script = '''
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]);value=json.loads(sys.argv[2])
state=json.loads((root/'status.json').read_text())
assert state['status']=='held_idle' and state.get('active_requests')==0,state
temp=root/'control.coordinator.tmp';temp.write_text(json.dumps(value,indent=2)+'\\n')
temp.replace(root/'control.json')
print(json.dumps({'admitted':value}))
'''
    return remote_json(HOST, script, (LOAD, json.dumps(control)))


def metrics(label, boundary):
    cfg = json.loads((ROOT / "deploy_config.json").read_text())
    key = Path(cfg["api_key_file"]).read_text().strip()
    request = urllib.request.Request(f"http://127.0.0.1:{cfg['port']}/metrics",
        headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(request, timeout=10) as response:
        (OUT / f"{label}-{boundary}.prom").write_bytes(response.read())


def main():
    plan = remote_json(HOST, "import pathlib,sys;print((pathlib.Path(sys.argv[1])/'comparison-plan.json').read_text())", (LOAD,))
    # The separately prepared plan may wrap its ordered cycle list.
    if isinstance(plan, dict):
        for key in ("cycles", "comparison_order", "order", "plan"):
            if isinstance(plan.get(key), list):
                plan = plan[key]
                break
    if not isinstance(plan, list) or len(plan) != 10:
        raise RuntimeError("Expected the agreed ten-cycle plan")
    expected = ["external", "in_process", "external", "in_process", "in_process",
                "external", "in_process", "external", "external", "in_process"]
    assert [row["backend"] for row in plan] == expected
    (OUT / "campaign-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    checkpoint = OUT / "campaign-status.json"
    complete = []
    for item in plan:
        state = status()
        assert state["status"] == "held_idle" and state.get("active_requests") == 0, state
        prior_cycles = state["cycles_started"]
        for attempt in range(30):
            try:
                idle()
                break
            except RuntimeError:
                time.sleep(1)
        else:
            raise RuntimeError("Server failed to drain")
        label = item["backend_epoch"]
        print("SWITCH", label, flush=True)
        subprocess.run([PYTHON, str(BASE / "set_backend.py"), "--mode", item["backend"],
                        "--label", label], check=True)
        metrics(label, "before")
        admission = {"action": "run_once", "profile": "c12", "backend_epoch": label}
        print(json.dumps(admit(admission)), flush=True)
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            time.sleep(10)
            state = status()
            checkpoint.write_text(json.dumps({"current": item, "client": state,
                "completed": complete, "updated_ns": time.time_ns()}, indent=2) + "\n")
            print("STATUS", label, state["status"], state.get("cycles_started"), flush=True)
            if state["status"] in ("stopped_after_request_error", "stopped"):
                raise RuntimeError(f"Client stopped: {state}")
            if state["cycles_started"] > prior_cycles and state["status"] == "held_idle":
                break
        else:
            raise TimeoutError(f"Cycle {label} did not finish")
        cycle = remote_json(HOST, "import json,pathlib,sys;lines=(pathlib.Path(sys.argv[1])/'cycles.jsonl').read_text().splitlines(); print(lines[-1])", (LOAD,))
        assert cycle["backend_epoch"] == label and cycle["status"] == "complete", cycle
        assert cycle["request_errors"] == 0 and cycle["requests_completed"] == 12, cycle
        assert not cycle["epoch_changed_during_cycle"] and not cycle["output_usage_missing"], cycle
        metrics(label, "after")
        complete.append(cycle)
        print("COMPLETE", label, "tokens/s", cycle["aggregate_output_tokens_per_second"], flush=True)
        subprocess.run([PYTHON, str(BASE / "collect.py")], check=True)
        subprocess.run([PYTHON, str(BASE / "analyze.py")], check=True)
        report = json.loads((OUT / "comparison.json").read_text())
        if report["audit_errors"]:
            raise RuntimeError(f"Comparison audit failed: {report['audit_errors'][:10]}")
    checkpoint.write_text(json.dumps({"state": "complete", "completed": complete,
                                      "updated_ns": time.time_ns()}, indent=2) + "\n")
    print("CAMPAIGN COMPLETE; client held idle; fixed in_process mode remains selected", flush=True)


if __name__ == "__main__":
    main()
