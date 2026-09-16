"""Actual ZMQ worker, actual native reader, shared output, and CUDA graph switches."""

import ctypes
import json
import multiprocessing as mp
import os
from pathlib import Path
import struct
import threading
import time
from types import SimpleNamespace as NS

import numpy as np
import torch
from torch import nn
import zmq

from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpNGramEmbedding as Qwen,
    Qwen4ExpPLEFp8EmbeddingMethod,
)
from vllm.v1.ple_offload.backend_comparison import PleBackendComparison
from vllm.v1.ple_offload.backend_control import encode, decode
from vllm.v1.ple_offload.worker import PleOffloadRunner, _init_offload_distributed


def configuration():
    text = NS(ngram_size=3, heads_per_ngram=8, eos_token_id=999,
              vocab_size=1000, seed=1234, ngram_vocab_size_base=101,
              make_ngram_vocab_size_divisible_by=128,
              ple_embedding_dtype="float8_e4m3fn", ple_embed_dim=2560)
    return NS(parallel_config=NS(nnodes=2, tensor_parallel_size=2,
                                data_parallel_size=1, rank=0, world_size=2),
              scheduler_config=NS(max_num_batched_tokens=128, max_num_seqs=32),
              model_config=NS(hf_text_config=text, dtype=torch.bfloat16))


def worker(ipc, ready, stop):
    from vllm.model_executor.layers.ple_offload_layer import mark_as_offload_worker
    mark_as_offload_worker()
    _init_offload_distributed()
    vc = configuration()
    layer = Qwen(vc.model_config.hf_text_config, 2560, 0, 128, 32, "layer", "layer")
    layer.layer_multipliers.copy_(torch.tensor([11, 23, 37]))
    layer.ngram_embedding.weight_scale.data.fill_(1.)
    PleOffloadRunner._attach_packed_table(
        "layer", layer, "/tmp/ple-test-table/layer.ngram_embedding.packed_u8"
    )
    # Use the production registration/dispatch/gather/publication implementation;
    # only the heavyweight checkpoint loader is replaced by this tiny table.
    runner = object.__new__(PleOffloadRunner)
    runner.vllm_config = vc
    runner._gb10_mapped = True
    runner._clamp_input_ids = True
    runner._layers = {"layer": layer}
    runner._worker_targets = {}
    runner._pinned_bufs = {}
    runner._input_bufs = {}
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.bind(ipc)
    ready.send("bound")
    runner.accept_registrations(sock, 1)
    ready.send("registered")
    handle = runner._handle_requests
    def delayed(requests):
        if any(r.seq % 17 == 0 for r in requests):
            time.sleep(.005)
        handle(requests)
    runner._handle_requests = delayed
    runner.busy_loop(sock, stop)
    sock.close()
    ctx.term()


def main():
    _init_offload_distributed()
    vc = configuration()
    config = vc.model_config.hf_text_config
    sizes, offsets, rows = Qwen._make_vocab_layout(
        ngram_vocab_size_base=101, ngram_heads=16, ple_dense_layer_id=0
    )
    rows = ((rows + 127) // 128) * 128
    root = Path("/tmp/ple-test-table")
    root.mkdir(exist_ok=True)
    table = torch.from_numpy(np.random.default_rng(7).integers(
        0, 256, (rows, 160), dtype=np.uint8
    ))
    path = root / "layer.ngram_embedding.packed_u8"
    path.write_bytes(table.numpy().tobytes())
    Path(str(path) + ".json").write_text(json.dumps({
        "row_format": "fp8_e4m3", "total_rows": rows, "row_width": 160,
    }))
    control = Path(os.environ["GB10_PLE_BACKEND_CONTROL"])
    control.write_bytes(struct.pack("<Q", encode("in_process", 1)) + bytes(4088))
    Path(os.environ["GB10_PLE_READ_CONTROL"]).write_bytes(bytes(4096))
    for mode in ("abba", "baab"):
        seq = [decode(encode(mode, 7, 2), i)["backend"] for i in range(1, 9)]
        assert seq == (["in_process"] * 2 + ["external"] * 4 + ["in_process"] * 2
                       if mode == "abba" else
                       ["external"] * 2 + ["in_process"] * 4 + ["external"] * 2)
    for invalid in (0, 255, encode("external", 1) | (1 << 24)):
        try:
            decode(invalid, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid control word accepted")
    methods = ("in_process", "external", "abba", "baab")
    context = mp.get_context("spawn")
    all_records = []
    for source_device in ("cpu", "cuda"):
        ipc = f"ipc:///tmp/ple-backend-test-{source_device}.sock"
        receive, send = context.Pipe(duplex=False)
        stop = context.Event()
        child = context.Process(target=worker, args=(ipc, send, stop))
        child.start()
        assert receive.poll(60), "External fixture failed to bind"
        assert receive.recv() == "bound"
        model = nn.Module()
        model.layer = Qwen(config, 2560, 0, 128, 32, "layer", "layer")
        model.layer._offload_quant_method = Qwen4ExpPLEFp8EmbeddingMethod()
        model.layer.load_weights(iter([
            ("layer_multipliers", torch.tensor([11, 23, 37])),
            ("ngram_heads_vocab_sizes", torch.tensor(sizes)),
            ("ngram_heads_offsets", torch.tensor(offsets)),
            ("ngram_embedding.weight_scale", torch.tensor(1.)),
        ]))
        inputs = torch.arange(128, dtype=torch.int32, device=source_device)
        query = torch.zeros(33, dtype=torch.int32, device=source_device)
        history = torch.zeros((32, 2), dtype=torch.int32, device=source_device)
        connector = PleBackendComparison(vc, model, torch.device("cuda:0"), ipc,
            input_ids_source=inputs, query_start_loc_source=query,
            ngram_context_source=history)
        assert receive.poll(60), "External registration failed"
        assert receive.recv() == "registered"
        output_address = model.layer._gpu_output_buffer.data_ptr()
        assert connector.external._done_flag.data_ptr() == connector.in_process._done_flag.data_ptr()
        assert connector.external.output_owners["layer"].data_ptr() == connector.in_process._owners[0].data_ptr()
        connector.signal_dummy_outputs(128)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual_graph = model.layer(torch.empty(0, device="cuda"), inputs).view(torch.uint8).clone()
        shapes = [(1, [4], 4, False), (12, [4]*12, 48, False),
                  (32, [4]*32, 128, True), (13, [4]*13, 64, False),
                  (2, [63,65], 128, False)]
        control_fd = os.open(control, os.O_WRONLY)
        try:
            for step in range(120):
                requests, lengths, tokens, full = shapes[step % len(shapes)]
                word = encode(methods[(step // 5) % 4], step // 5 + 1, 2)
                os.pwrite(control_fd, struct.pack("<Q", word), 0)
                host_inputs = (torch.arange(128, dtype=torch.int32) + step * 7) % 997
                host_inputs[step % 128] = -1
                host_inputs[(step + 2) % 128] = 999
                host_history = torch.arange(64, dtype=torch.int32).reshape(32, 2) + step
                host_history[0, 0] = 999
                host_query = torch.tensor([0] + list(np.cumsum(lengths)), dtype=torch.int32)
                connector.start_step()
                inputs.copy_(host_inputs)
                query[:requests+1].copy_(host_query)
                history.copy_(host_history)
                batch = NS(num_reqs=requests, num_tokens=sum(lengths),
                    num_tokens_after_padding=tokens, has_prefill=max(lengths)>4,
                    num_scheduled_tokens=np.array(lengths),
                    seq_lens_cpu_upper_bound=torch.tensor(lengths)+100,
                    is_prefilling_np=np.array([n>4 for n in lengths]),
                    num_draft_tokens=0, num_draft_tokens_per_req=None)
                connector.record_batch(batch, full)
                deferred = full and connector.supports_deferred_completion
                connector.prepare_forward(requests, tokens, False, defer_completion=deferred)
                if full:
                    graph.replay()
                    actual = actual_graph
                else:
                    actual = model.layer(torch.empty(0, device="cuda"), inputs[:tokens]).view(torch.uint8).clone()
                if deferred:
                    connector.finish_forward()
                connector.release_outputs()
                copy_event = torch.cuda.Event()
                copy_event.record()
                connector.end_step(NS(copy_event=copy_event,
                    num_sampled_tokens_np=np.ones(requests, dtype=np.int32)))
                ids = torch.empty((tokens,16), dtype=torch.int64)
                hs = connector.in_process._hash_state[0]
                cpu_tokens = host_inputs[:tokens].to(torch.int64).clamp_min_(0)
                cpu_query = host_query.to(torch.int64)
                cpu_history = host_history[:requests].to(torch.int64)
                code = connector.in_process._hash(cpu_tokens.data_ptr(), tokens,
                    cpu_query.data_ptr(), requests, cpu_history.data_ptr(), 2,
                    3, 8, 999, *(v.data_ptr() for v in hs[3]), ids.data_ptr())
                assert code == 0
                expected = table.index_select(0, ids.flatten()).reshape(tokens,2560)
                assert torch.equal(actual.cpu(), expected), (source_device, step)
                assert model.layer._gpu_output_buffer.data_ptr() == output_address
                assert int(connector._done_flag[0]) == step + 1
                assert child.is_alive()
            connector.close()
            connector.close()
        finally:
            os.close(control_fd)
            stop.set()
            child.join(20)
            if child.is_alive():
                child.terminate()
                child.join(10)
        assert child.exitcode == 0, child.exitcode
        records = [json.loads(line) for line in connector.log_path.read_text().splitlines()]
        steps = [r for r in records if r["event"] == "step"]
        assert len(steps) == 120
        assert {r["backend"] for r in steps} == {"external", "in_process"}
        assert all(r["gpu_iteration_ms"] > 0 for r in steps)
        assert all(r.get("gpu_wait_error", 0) == 0 for r in steps)
        assert max(r.get("gpu_wait_us", 0) for r in steps if r["backend"] == "external") > 1000
        all_records.extend(steps)
        print("PASS", source_device, "120 real IPC/native switches; eager/FULL; padding/EOS/negative IDs; exact GPU bytes; delayed publication", flush=True)
    print("PASS total", len(all_records), "steps; stable captured pointers and monotonic completion sequences", flush=True)


if __name__ == "__main__":
    main()
