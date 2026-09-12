"""Small exactness/performance grid through the unchanged public operator."""

import json
import statistics
import sys
import torch
import vllm._custom_ops  # noqa: F401

from test_top_k_per_row import _exact_topk_reference

arm = sys.argv[1]
torch.manual_seed(55122)
for rows, width, length, k in [
    (1, 8192, 8192, 512),
    (32, 8192, 8192, 512),
    (1, 32768, 32768, 512),
    (64, 32768, 32768, 512),
    (1, 163840, 8192, 512),
    (128, 163840, 8192, 512),
    (1, 65536, 65536, 2048),
    (64, 40000, 40000, 2048),
]:
    logits = 1.0 + 1e-3 * torch.randn(rows, width, device="cuda")
    lengths = torch.full((rows,), length, device="cuda", dtype=torch.int32)
    out = torch.empty((rows, k), device="cuda", dtype=torch.int32)
    ws = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)
    ref = _exact_topk_reference(logits, lengths, k)

    def call():
        torch.ops._C.persistent_topk(logits, lengths, out, ws, k, width)

    copies = []
    for _ in range(6):
        call()
        copies.append(out.clone())
    torch.cuda.synchronize()
    valid = (copies[0] >= 0) & (copies[0] < length)
    kth = logits[:, :length].topk(k, dim=1).values[:, -1:]
    values = logits.gather(1, copies[0].long().clamp(0, width - 1))
    subthreshold = int(((values < kth) & valid).sum())
    sets = [c.sort(dim=1).values for c in copies]
    timing = []
    # Graph batching removes Python launch cost; external GPU load is still present.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(20):
            call()
    for _ in range(10):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        timing.append(start.elapsed_time(end) * 1000 / 20)
    print(
        json.dumps(
            dict(
                arm=arm,
                rows=rows,
                width=width,
                length=length,
                k=k,
                exact=torch.equal(copies[0], ref),
                repeatable=all(torch.equal(copies[0], c) for c in copies),
                sets_repeatable=all(torch.equal(sets[0], c) for c in sets),
                subthreshold_selected=subthreshold,
                invalid_selected=int((~valid).sum()),
                median_us=statistics.median(timing),
                min_us=min(timing),
            )
        ),
        flush=True,
    )
