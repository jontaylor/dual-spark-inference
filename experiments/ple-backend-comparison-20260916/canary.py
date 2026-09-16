"""Small functional inference checks; does not assert model quality/performance."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
cfg = json.loads((ROOT / "deploy_config.json").read_text())
key = Path(cfg["api_key_file"]).read_text().strip()
URL = f"http://127.0.0.1:{cfg['port']}"


def call(path, body=None):
    req = urllib.request.Request(URL + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read())


def inference(pair):
    prompt, expected = pair
    result = call("/v1/chat/completions", {
        "model": cfg["served_names"][0], "temperature": 0, "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}],
    })
    content = result["choices"][0]["message"]["content"].strip()
    if expected is not None:
        assert content == expected, content
    assert result["usage"]["completion_tokens"] > 0
    return {"content": content, "usage": result["usage"],
            "finish_reason": result["choices"][0]["finish_reason"]}


results = {"models": [r["id"] for r in call("/v1/models")["data"]], "backends": {}}
for backend in ("in_process", "external", "in_process"):
    for attempt in range(30):
        result = subprocess.run([str(ROOT / ".venv/bin/python"), str(BASE / "set_backend.py"),
            "--mode", backend, "--label", "canary-" + backend], capture_output=True, text=True)
        if result.returncode == 0:
            break
        time.sleep(1)
    else:
        raise RuntimeError(result.stderr)
    prompts = [("Reply with exactly READY and nothing else.", "READY"),
               ("What is 7 + 8? Reply with only the number.", "15"),
               ("Count from 1 to 30, separated by commas. No other text.", None)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        rows = list(executor.map(inference, prompts))
    results["backends"].setdefault(backend, []).append(rows)
    print("PASS functional requests", backend, flush=True)
(ROOT / "results/ple-backend-comparison-20260916/canary.json").write_text(
    json.dumps(results, indent=2) + "\n")
