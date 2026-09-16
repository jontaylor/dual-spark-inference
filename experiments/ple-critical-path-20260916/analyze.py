"""Recompute descriptive PLE critical-path evidence from existing captures.

No serving changes, live probes, GPU work, or benchmark traffic.
"""

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
CAPTURES = (
    "ple-latency-campaign-20260916/final-stability",
    "ple-resident-20260916/live-ab",
)
FIELDS = (
    "reader_ms", "submit_ms", "gpu_wait_us", "gpu_hash_to_gate_ms",
    "gpu_launch_to_expected_ms", "next_period_ms",
)


def quantile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)]
                            - values[lower]) * (position - lower)


def summarize(rows):
    result = {"batches": len(rows)}
    for field in FIELDS:
        values = [float(row[field]) for row in rows if row.get(field)]
        if values:
            result[field] = {
                "samples": len(values), "mean": mean(values),
                "p50": quantile(values, .5), "p95": quantile(values, .95),
                "p99": quantile(values, .99), "maximum": max(values),
                "sum": sum(values),
            }
    waits = [row for row in rows if row.get("gpu_wait_polls")]
    result["batches_with_wait_poll"] = sum(
        float(row["gpu_wait_polls"]) > 0 for row in waits
    )
    return result


def main():
    output = {
        "interpretation": (
            "Descriptive historical evidence, not a before/after estimate. "
            "GPU wait measures the wait-kernel body only. Hash-to-gate includes "
            "intervening work. next_period_ms is a CPU step proxy. "
            "Synchronous includes eager/piecewise, not exclusively prefill. "
            "In live-ab, inline means prior SQPOLL and async means prefetch."
        ),
        "captures": {},
    }
    for capture in CAPTURES:
        ranks = []
        rank_results = {}
        for rank in (0, 1):
            path = ROOT / "results" / capture / f"rank{rank}-steps.csv"
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
            ranks.append({int(row["seq"]): row for row in rows})
            rank_results[str(rank)] = {
                mode: {
                    policy: summarize([
                        row for row in rows
                        if row["deferred"] == deferred and row["policy"] == policy
                    ])
                    for policy in sorted({row["policy"] for row in rows
                                          if row["deferred"] == deferred})
                }
                for mode, deferred in (("full_graph", "1"), ("synchronous", "0"))
            }
        common = sorted(ranks[0].keys() & ranks[1].keys())
        decode = [seq for seq in common if all(
            ranks[rank][seq]["deferred"] == "1"
            and ranks[rank][seq].get("gpu_wait_us") for rank in (0, 1)
        )]
        max_waits = [max(float(ranks[rank][seq]["gpu_wait_us"])
                         for rank in (0, 1)) for seq in decode]
        output["captures"][capture] = {
            "ranks": rank_results,
            "aligned_decode": {
                "batches": len(decode),
                "mean_rank_max_wait_us": mean(max_waits) if max_waits else None,
                "sum_rank_max_wait_ms": sum(max_waits) / 1000,
                "note": "Rank maximum is descriptive, not a TP critical-path estimator.",
            },
        }
    destination = Path(__file__).with_name("evidence.json")
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
