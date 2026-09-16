"""Audit a completed fixed-SQPOLL capture without ABBA assumptions."""
import collections
import csv
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
metadata = json.loads((root / "capture.json").read_text())
summary = json.loads((root / "summary.json").read_text())
assert metadata["async_policy"] == "sqpoll"
assert metadata["end"] - metadata["start"] >= 1200
ranks = {}
steps = {}
for rank in (0, 1):
    assert not (root / f"rank{rank}.err").read_text().strip(), "BPF warnings/errors"
    assert summary[str(rank)]["errors"] == 0, "native reader error"
    with (root / f"rank{rank}-steps.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows, "no matched steps"
    assert len({r["seq"] for r in rows}) == len(rows), "duplicate sequence"
    wrong = [r["seq"] for r in rows if r["policy"] != (
        "sqpoll" if r["deferred"] == "1" else "inline")]
    assert not wrong, (rank, "unexpected policy", wrong[:5])
    steps[rank] = {int(r["seq"]): r for r in rows}
    decode = [r for r in rows if r["deferred"] == "1"]
    assert decode, "no full-graph batches"
    strata = collections.Counter((r["requests"], r["rows"]) for r in decode)
    ranks[rank] = {
        "reader_batches": summary[str(rank)]["all"]["n"],
        "matched_steps": len(rows),
        "native_errors": summary[str(rank)]["errors"],
        "policies": dict(collections.Counter(r["policy"] for r in rows)),
        "decode_strata": {f"r{req}/rows{count}": n for (req, count), n in strata.items()},
        "decode": summary[str(rank)]["decode"],
    }
common = sorted(steps[0].keys() & steps[1].keys())
assert common, "no aligned ranks"
keys = ("rows", "tokens", "requests", "policy", "deferred")
mismatches = [s for s in common if any(steps[0][s][k] != steps[1][s][k] for k in keys)]
assert not mismatches, ("rank shape/policy mismatch", mismatches[:5])
result = {
    "duration_seconds": metadata["end"] - metadata["start"],
    "pids": [r["pid"] for r in metadata["ranks"]],
    "common_sequences": len(common),
    "rank_mismatches": len(mismatches),
    "ranks": ranks,
    "status": "PASS",
}
(root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
