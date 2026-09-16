"""Recompute final cycle aggregates and telemetry coverage without live changes."""

from collections import Counter
import json
from pathlib import Path
import statistics

from analyze import summarize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/ple-backend-comparison-20260916"


def metric_totals(path):
    totals = Counter()
    for line in path.read_text().splitlines():
        if not line.startswith("vllm:"):
            continue
        series, value = line.rsplit(" ", 1)
        name = series.split("{", 1)[0]
        if name in ("vllm:request_success_total", "vllm:generation_tokens_total",
                    "vllm:prompt_tokens_total", "vllm:prompt_tokens_cached_total",
                    "vllm:num_requests_running", "vllm:num_requests_waiting"):
            totals[name] += float(value)
    return dict(totals)


def main():
    campaign = json.loads((OUT / "campaign-status.json").read_text())
    comparison = json.loads((OUT / "comparison.json").read_text())
    client = json.loads((OUT / "client.json").read_text())
    assert campaign.get("state") == "complete", "Campaign is not complete"
    assert not comparison["audit_errors"], comparison["audit_errors"]
    cycles = comparison["measured_cycles"]
    assert len(cycles) == 8 and len(comparison["pairs"]) == 4
    assert Counter(c["backend"] for c in cycles) == {"external": 4, "in_process": 4}
    assert {c["output_tokens"] for c in cycles} == {24576}
    assert {c["requests_completed"] for c in cycles} == {12}
    assert {c["uncached_prompt_tokens"] for c in cycles} == {42302}
    assert not any(c["cache_usage_missing"] for c in cycles)
    epochs = {c["epoch"]: c["backend"] for c in cycles}
    labels = {c["label"]: c["backend"] for c in cycles}
    result = {"estimate": comparison["estimate"], "pairs": comparison["pairs"],
              "measured_requests": 96, "measured_output_tokens": 196608,
              "uncached_prompt_tokens_per_cycle": 42302,
              "backends": {}, "telemetry": {}, "server_metric_deltas": {}}
    for cycle in cycles:
        label = cycle["label"]
        before = metric_totals(OUT / f"{label}-before.prom")
        after = metric_totals(OUT / f"{label}-after.prom")
        for boundary in (before, after):
            assert boundary["vllm:num_requests_running"] == 0
            assert boundary["vllm:num_requests_waiting"] == 0
        delta = {key: after[key] - before[key] for key in before}
        assert delta["vllm:request_success_total"] == 12
        assert delta["vllm:generation_tokens_total"] == 24576
        assert delta["vllm:prompt_tokens_total"] == 180542
        assert delta["vllm:prompt_tokens_cached_total"] == 138240
        result["server_metric_deltas"][label] = delta
    for backend in ("external", "in_process"):
        arm = [c for c in cycles if c["backend"] == backend]
        requests = [r for r in client["requests.jsonl"]
                    if r["status"] == "complete"
                    and labels.get(r["backend_epoch"]) == backend]
        assert len(requests) == 48
        for request in requests:
            assert request["output_tokens"] == 2048
            request["tpot_ms"] = request["generation_span_seconds"] * 1000 / 2047
        seconds = sum(c["elapsed_seconds"] for c in arm)
        tokens = sum(c["output_tokens"] for c in arm)
        result["backends"][backend] = {
            "cycles": 4, "requests": len(requests), "output_tokens": tokens,
            "total_cycle_seconds": seconds,
            "pooled_output_tokens_per_second": tokens / seconds,
            "mean_cycle_output_tokens_per_second": statistics.mean(
                c["aggregate_output_tokens_per_second"] for c in arm),
            "client_ttft_seconds": summarize(requests, "ttft_seconds"),
            "client_tpot_ms": summarize(requests, "tpot_ms"),
        }
    ranks = []
    for rank in (0, 1):
        rows = [json.loads(line)
                for path in sorted((OUT / f"live/rank{rank}").glob("rank*.jsonl"))
                for line in path.read_text().splitlines()]
        begins = [r["seq"] for r in rows if r["event"] == "begin"]
        steps = [r for r in rows if r["event"] == "step"]
        seqs = [r["seq"] for r in steps]
        assert begins == list(range(1, max(begins) + 1))
        assert seqs == list(range(1, max(seqs) + 1))
        assert not any(r.get("gpu_wait_error", 0) for r in steps)
        ranks.append({r["seq"]: r for r in steps})
        result["telemetry"][f"rank{rank}"] = {
            "begun_steps": len(begins), "completed_steps": len(steps),
            "pending_tail_sequences": sorted(set(begins) - set(seqs)),
            "contiguous_sequences": True,
        }
    common = set(ranks[0]) & set(ranks[1])
    for rank in (0, 1):
        for backend in ("external", "in_process"):
            rows = [ranks[rank][seq] for seq in sorted(common)
                    if epochs.get(ranks[rank][seq]["epoch"]) == backend]
            groups = {
                "all": rows,
                "full_decode": [r for r in rows if r["full_graph"] and not r["has_prefill"]],
                "full_decode_12": [r for r in rows if r["full_graph"]
                                   and not r["has_prefill"] and r["requests"] == 12],
                "has_prefill": [r for r in rows if r["has_prefill"]],
                "non_full_without_prefill": [r for r in rows if not r["full_graph"]
                                             and not r["has_prefill"]],
            }
            summary = {}
            for name, group in groups.items():
                waits = [r for r in group if "gpu_wait_us" in r]
                summary[name] = {
                    "steps": len(group),
                    **{field: summarize(group, field) for field in (
                        "gpu_iteration_ms", "gpu_wait_us", "prepare_cpu_ms",
                        "consumer_fence_ms", "complete_cpu_ms", "cpu_enqueue_ms")},
                    "wait_samples": len(waits),
                    "steps_with_wait_polls": sum(r.get("gpu_wait_polls", 0) > 0 for r in waits),
                    "maximum_wait_us": max((r["gpu_wait_us"] for r in waits), default=None),
                }
            summary["scheduled_drafts"] = sum(r["scheduled_draft_tokens"] for r in rows)
            summary["accepted_drafts"] = sum(r["accepted_draft_tokens"] for r in rows)
            summary["acceptance_fraction"] = summary["accepted_drafts"] / summary["scheduled_drafts"]
            result["telemetry"][f"rank{rank}"][backend] = summary
    destination = OUT / "final-summary.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "estimate": result["estimate"],
                      "backends": result["backends"]}, indent=2))


if __name__ == "__main__":
    main()
