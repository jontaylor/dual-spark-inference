"""Deployment-specific checks for the public top-k op and native-library replacement."""

import pytest
import torch
import vllm._custom_ops  # noqa: F401

from test_top_k_per_row import _exact_topk_reference


@pytest.mark.parametrize(
    "rows,width,k",
    [
        (1, 163840, 512),
        (32, 163840, 512),
        (128, 163840, 512),
        (64, 65536, 1024),
        (64, 65536, 2048),
        (1, 400000, 512),
    ],
)
def test_paged_capacity_graph_replay(rows, width, k):
    """Replaying a captured launch must use new ragged lengths and score values."""
    gen = torch.Generator(device="cuda").manual_seed(55122)
    scores = torch.randn(rows, width, device="cuda", generator=gen)
    lengths = torch.full((rows,), 256, dtype=torch.int32, device="cuda")
    output = torch.empty(rows, k, dtype=torch.int32, device="cuda")
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda")

    def call():
        torch.ops._C.persistent_topk(scores, lengths, output, workspace, k, width)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    for length in (0, 700, 8192, 22016, 22017, 32768, width):
        lengths.fill_(length)
        if rows > 1:
            lengths[0] = 0
            lengths[-1] = max(0, length - 7)
        scores.copy_(torch.randint(0, 8, scores.shape, device="cuda", generator=gen))
        expected = _exact_topk_reference(scores, lengths, k)
        for _ in range(4):
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output, expected), (rows, width, k, length)


def test_empty_batch():
    torch.ops._C.persistent_topk(
        torch.empty(0, 512, device="cuda"),
        torch.empty(0, dtype=torch.int32, device="cuda"),
        torch.empty(0, 512, dtype=torch.int32, device="cuda"),
        torch.empty(1024 * 1024, dtype=torch.uint8, device="cuda"),
        512,
        512,
    )
    torch.cuda.synchronize()


@pytest.mark.parametrize("rows,pages_per_request", [(4, 64), (64, 313)])
def test_paged_qsa_selection_matches_exact_reference(rows, pages_per_request):
    """The deployed scorer/selector/expander must consume the exact block set."""
    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        expand_qsa_block_indices_cuda,
        qsa_mqa_paged,
        qsa_select_paged_tokens,
    )

    page_size, dim, ratio, token_topk = 128, 128, 4, 2048
    torch.manual_seed(55122)
    q = torch.randn(rows, 8, dim, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(
        2 * pages_per_request, page_size, 1, dim, device="cuda", dtype=torch.bfloat16
    )
    table = torch.randperm(
        2 * pages_per_request, device="cuda", dtype=torch.int32
    ).reshape(2, -1)
    mapping = torch.arange(rows, device="cuda", dtype=torch.int32) % 2
    lengths = torch.tensor(
        [
            pages_per_request * page_size * ratio - 7,
            pages_per_request * page_size * ratio - 19,
        ],
        device="cuda",
        dtype=torch.int32,
    )
    positions = (
        lengths[mapping.long()]
        - 1
        - torch.arange(rows, device="cuda", dtype=torch.int32) % 4
    )
    scores, visible = qsa_mqa_paged(q, cache, table, mapping, positions, lengths, ratio)
    selected = _exact_topk_reference(scores, visible, token_topk // ratio)
    expected = expand_qsa_block_indices_cuda(
        selected, positions, lengths, mapping, ratio, token_topk
    )
    for _ in range(4):
        actual = qsa_select_paged_tokens(
            q, cache, table, mapping, positions, lengths, token_topk, ratio
        )
        torch.cuda.synchronize()
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_activation_smoke(dtype):
    x = torch.randn(16, 256, dtype=dtype, device="cuda")
    out = torch.empty(16, 128, dtype=dtype, device="cuda")
    torch.ops._C.silu_and_mul(out, x)
    expected = (torch.nn.functional.silu(x[:, :128].float()) * x[:, 128:].float()).to(
        dtype
    )
    torch.testing.assert_close(
        out,
        expected,
        atol=0.02 if dtype == torch.bfloat16 else 1e-5,
        rtol=0.02 if dtype == torch.bfloat16 else 1e-5,
    )
