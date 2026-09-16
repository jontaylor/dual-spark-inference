"""Cycle-level useful throughput and common per-step GPU timing, with audit."""

from collections import defaultdict
import json
import math
from pathlib import Path
import statistics as stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/ple-backend-comparison-20260916"


def quantile(values, fraction):
    values = sorted(values)
    pos = (len(values) - 1) * fraction
    index = int(pos)
    return values[index] + (values[min(index + 1, len(values) - 1)] - values[index]) * (pos - index)


def summarize(rows, field):
    values = [r[field] for r in rows if r.get(field) is not None]
    return ({"n": len(values), "mean": stats.mean(values),
             "p50": quantile(values, .5), "p95": quantile(values, .95)}
            if values else {})


def main():
    controls = [json.loads(line) for line in (OUT / "control.jsonl").read_text().splitlines()]
    by_label = {row["label"]: row for row in controls}
    client = json.loads((OUT / "client.json").read_text())
    cycles = client.get("cycles.jsonl", [])
    measured = [r for r in cycles if r["backend_epoch"].startswith("ple-measured-")]
    ranks = []
    errors = []
    for rank in (0, 1):
        records = []
        files = sorted((OUT / f"live/rank{rank}").glob("rank*.jsonl"))
        for path in files:
            for line in path.read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    errors.append(f"rank{rank}: incomplete JSON record")
        boots = {r["boot"] for r in records if r["event"] == "init"}
        if len(boots) != 1:
            errors.append(f"rank{rank}: expected one worker lifetime, got {len(boots)}")
        begins = [r for r in records if r["event"] == "begin"]
        steps = [r for r in records if r["event"] == "step"]
        if len({r["seq"] for r in begins}) != len(begins):
            errors.append(f"rank{rank}: duplicate begin sequences")
        if len({r["seq"] for r in steps}) != len(steps):
            errors.append(f"rank{rank}: duplicate completed sequences")
        if any(r.get("gpu_wait_error", 0) for r in steps):
            errors.append(f"rank{rank}: GPU wait error")
        ranks.append({r["seq"]: r for r in steps})
    common = sorted(ranks[0].keys() & ranks[1].keys())
    for seq in common:
        for field in ("backend", "epoch", "requests", "tokens", "padded_tokens",
                      "full_graph", "has_prefill", "scheduled_draft_tokens",
                      "accepted_draft_tokens", "sampled_tokens_before_stop_filter"):
            if ranks[0][seq][field] != ranks[1][seq][field]:
                errors.append(f"seq {seq}: rank disagreement in {field}")
    cycle_results = []
    signatures = {(r["config_sha256"], r["prompts_sha256"], r["concurrency"],
                   r["requests_per_lane"]) for r in measured}
    if len(signatures) > 1:
        errors.append("Measured workload settings changed between cycles")
    for cycle in measured:
        label = cycle["backend_epoch"]
        if label not in by_label:
            errors.append(f"{label}: no recorded backend switch")
            continue
        control = by_label[label]
        if (cycle["status"] != "complete" or cycle["request_errors"]
                or cycle["output_usage_missing"] or cycle["epoch_changed_during_cycle"]):
            errors.append(f"{label}: incomplete/error/mixed client cycle")
        row = {"label": label, "backend": control["backend"], "epoch": control["epoch"],
               **{key: cycle[key] for key in (
                   "elapsed_seconds", "output_tokens", "aggregate_output_tokens_per_second",
                   "uncached_prompt_tokens", "requests_completed", "cache_usage_missing")}}
        requests = [r for r in client.get("requests.jsonl", [])
                    if r["backend_epoch"] == label and r["status"] == "complete"]
        for request in requests:
            n = request.get("output_tokens")
            span = request.get("generation_span_seconds")
            if n and n > 1 and span is not None:
                request["output_token_seconds"] = span / (n - 1)
        row["client_ttft_seconds"] = summarize(requests, "ttft_seconds")
        row["client_output_token_seconds"] = summarize(requests, "output_token_seconds")
        for rank in (0, 1):
            matched = [ranks[rank][seq] for seq in common
                       if ranks[rank][seq]["epoch"] == control["epoch"]]
            if any(r["backend"] != control["backend"] for r in matched):
                errors.append(f"{label}: actual backend mismatch")
            full = [r for r in matched if r["full_graph"] and not r["has_prefill"]]
            pure12 = [r for r in full if r["requests"] == 12]
            row[f"rank{rank}"] = {"aligned_steps": len(matched),
                "full_decode": {field: summarize(full, field) for field in (
                    "gpu_iteration_ms", "gpu_through_forward_ms", "gpu_wait_us", "prepare_cpu_ms")},
                "full_decode_12": {field: summarize(pure12, field) for field in (
                    "gpu_iteration_ms", "gpu_wait_us", "prepare_cpu_ms")},
                "scheduled_drafts": sum(r["scheduled_draft_tokens"] for r in matched),
                "accepted_drafts": sum(r["accepted_draft_tokens"] for r in matched)}
        cycle_results.append(row)
    pairs = []
    for index in range(0, len(cycle_results) - 1, 2):
        pair = cycle_results[index:index+2]
        mapping = {r["backend"]: r for r in pair}
        if set(mapping) != {"external", "in_process"}:
            errors.append("Measured adjacent pair does not contain both backends")
            continue
        ratio = (mapping["in_process"]["aggregate_output_tokens_per_second"] /
                 mapping["external"]["aggregate_output_tokens_per_second"])
        pairs.append({"labels": [r["label"] for r in pair],
                      "in_process_throughput_change_percent": (ratio - 1) * 100,
                      "log_ratio": math.log(ratio)})
    estimate = None
    if len(pairs) == 4:
        logs = [p["log_ratio"] for p in pairs]
        centre = stats.mean(logs)
        half = 3.182446305 * stats.stdev(logs) / math.sqrt(4)
        estimate = {"paired_geometric_throughput_change_percent": 100 * math.expm1(centre),
                    "approximate_95_percent_interval": [100 * math.expm1(centre - half),
                                                        100 * math.expm1(centre + half)],
                    "note": "Four adjacent cycle pairs; Student-t interval on log ratios, conditional on repeatability/independence; not per-token replication."}
    result = {"audit_errors": errors, "aligned_steps": len(common),
              "measured_cycles": cycle_results, "pairs": pairs, "estimate": estimate,
              "limitations": ["Synthetic c12 workload; both caches resident in one deployment.",
                  "GPU timing groups are descriptive, not context/acceptance-matched causal estimates.",
                  "Client cycle throughput includes queue, prefill and final concurrency drain.",
                  "Four pairs provide a preliminary estimate; workload trajectory differences must be checked."]}
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["PLE backend comparison", "", f"Aligned completed steps: {len(common)}; audit errors: {len(errors)}.", "",
             "| Cycle | Backend | Output tokens | Seconds | Tokens/s |",
             "|---|---|---:|---:|---:|"]
    for row in cycle_results:
        lines.append(f"| {row['label']} | {row['backend']} | {row['output_tokens']} | {row['elapsed_seconds']:.2f} | {row['aggregate_output_tokens_per_second']:.2f} |")
    if estimate:
        lines += ["", json.dumps(estimate, indent=2)]
    if errors:
        lines += ["", "Audit errors:", *["- " + error for error in errors]]
    lines += ["", *result["limitations"]]
    (OUT / "latest.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"cycles": len(cycle_results), "aligned_steps": len(common),
                      "audit_errors": errors[:10], "estimate": estimate}, indent=2))


if __name__ == "__main__":
    main()
