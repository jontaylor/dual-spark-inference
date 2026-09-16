# SPDX-License-Identifier: Apache-2.0
"""Node-local PLE in the GPU worker: one caller, batch reads, no IPC worker."""

import ctypes
import inspect
import json
import os
from pathlib import Path

import torch

from vllm.model_executor.layers.ple_offload_layer import (
    CpuGpuSemaphore,
    PleOffloadLayer,
)
from vllm.v1.ple_offload.connector import PleOffloadConnector
from vllm.v1.ple_offload.gb10_mapped import (
    map_shared_tensor,
    publish_done,
    publish_expected,
)
from vllm.v1.ple_offload.protocol import PleOffloadRequest

# A timed-out kernel read may still own native scratch space. Retain its handle
# until process exit if it cannot be reaped at close; never free a live target.
_pending_readers = []


class BatchReader:
    def __init__(self, path, rows, width, capacity, cache_bytes):
        self.lib = ctypes.CDLL("/opt/gb10/libple_batch_reader.so")
        v, i, u = ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint64
        self.lib.gb10_ple_reader_create.argtypes = [
            ctypes.c_char_p,
            i,
            i,
            i,
            u,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.gb10_ple_reader_create.restype = v
        self.lib.gb10_ple_reader_gather.argtypes = [v, v, i, v, ctypes.c_uint]
        self.lib.gb10_ple_reader_gather.restype = ctypes.c_int
        self.lib.gb10_ple_reader_destroy.argtypes = [v]
        self.lib.gb10_ple_reader_destroy.restype = ctypes.c_int
        error = ctypes.c_int()
        self.handle = self.lib.gb10_ple_reader_create(
            os.fsencode(path), rows, width, capacity, cache_bytes, ctypes.byref(error)
        )
        if not self.handle:
            raise OSError(
                -error.value,
                "PLE async reader initialization failed; "
                "check io_uring permissions and memory limits",
            )

    def gather(self, ids, output, timeout_ms):
        code = self.lib.gb10_ple_reader_gather(
            self.handle, ids.data_ptr(), ids.numel(), output.data_ptr(), timeout_ms
        )
        if code:
            raise OSError(-code, "PLE batch read failed; output not published")

    def close(self):
        if self.handle:
            code = self.lib.gb10_ple_reader_destroy(self.handle)
            if code:
                _pending_readers.append((self.lib, self.handle))
            self.handle = None


class PleInProcessConnector(PleOffloadConnector):
    """Reuse input staging and consumption fences, never construct IPC transport."""

    def __init__(
        self,
        vllm_config,
        model,
        device,
        ipc_addr,
        *,
        input_ids_source,
        query_start_loc_source,
        ngram_context_source,
    ):
        del ipc_addr
        pc = vllm_config.parallel_config
        if (
            os.environ.get("GB10_PLE_MAPPED_TRANSPORT") != "1"
            or os.environ.get("VLLM_PLE_LOCAL_TP") != "1"
            or pc.nnodes != 2
            or pc.tensor_parallel_size != 2
            or pc.data_parallel_size != 1
        ):
            raise ValueError("In-process PLE requires mapped node-local TP2/DP1")
        self.device = device
        self._gb10_mapped = True
        self._gb10_event_recorded = False
        self._gb10_consumed_event = torch.cuda.Event()
        self._seq = 0
        self._closed = False
        self._readers = []
        self._owners = []
        self._hash_state = []
        self._pinned_input_buffers = []
        self._layers = {
            name: layer
            for name, layer in model.named_modules()
            if isinstance(layer, PleOffloadLayer)
        }
        if len(self._layers) != 1:
            raise ValueError(
                "In-process batch reader currently requires exactly one PLE layer"
            )
        cfg = vllm_config.scheduler_config
        cache_mb = int(os.environ.get("GB10_PLE_ROW_CACHE_MB", "0"))
        if not 0 <= cache_mb <= 1024:
            raise ValueError("PLE row-cache budget must be between 0 and 1024 MiB")
        self._input_ids_source = input_ids_source
        self._query_start_loc_source = query_start_loc_source
        self._ngram_context_source = ngram_context_source
        self._uses_cuda_inputs = input_ids_source.is_cuda
        self._input_ids_buf = torch.empty(
            cfg.max_num_batched_tokens, dtype=torch.int32, device="cpu"
        ).share_memory_()
        self._query_start_loc_buf = torch.empty(
            cfg.max_num_seqs + 1, dtype=torch.int32, device="cpu"
        ).share_memory_()
        ngram = int(vllm_config.model_config.hf_text_config.ngram_size)
        self._ngram_context_buf = torch.empty(
            (cfg.max_num_seqs, ngram - 1), dtype=torch.int32, device="cpu"
        ).share_memory_()
        self._validate_input_sources()
        self._done_flag = torch.zeros(
            32, dtype=torch.int64, device="cpu"
        ).share_memory_()
        self._d2h_stream = self._input_ready_event = self._d2h_done_event = None
        self._timeout_ms = int(
            float(os.environ.get("VLLM_PLE_OFFLOAD_STEP_TIMEOUT", "30")) * 1000
        )
        if not 0 < self._timeout_ms <= 3600000:
            raise ValueError("PLE timeout must be in (0, 3600] seconds")
        native = ctypes.CDLL("/opt/gb10/libple_gather.so")
        self._native = native
        self._hash = native.gb10_ple_hash
        v, i = ctypes.c_void_p, ctypes.c_int64
        self._hash.argtypes = [v, i, v, i, v, i, i, i, i, v, v, v, v]
        self._hash.restype = ctypes.c_int
        try:
            self._pin_input_buffers()
            if self._uses_cuda_inputs:
                self._d2h_stream = torch.cuda.Stream(device=device)
                self._input_ready_event = torch.cuda.Event()
                self._d2h_done_event = torch.cuda.Event()
            mapped_flag = map_shared_tensor(self._done_flag, device)
            for name, layer in self._layers.items():
                args, kwargs = layer._inprocess_constructor
                bound = inspect.signature(type(layer).__init__).bind(
                    layer, *args, **kwargs
                )
                bound.apply_defaults()
                spec = bound.arguments
                config = spec["config"]
                heads = int(config.heads_per_ngram)
                count = (ngram - 1) * heads
                if ngram < 2 or ngram > 8 or not 1 <= heads <= 128:
                    raise ValueError("Unsupported native PLE hash dimensions")
                width = int(spec["embedding_dim"]) // count
                if width * count != int(spec["embedding_dim"]):
                    raise ValueError("Invalid PLE embedding width")
                dtype = layer.get_offload_output_dtype(vllm_config.model_config.dtype)
                if dtype != torch.float8_e4m3fn:
                    raise ValueError(
                        "In-process PLE currently supports packed FP8 rows only"
                    )
                dense_id = spec["ple_dense_layer_id"]
                multipliers = type(layer)._make_layer_multipliers(
                    ngram_size=ngram,
                    unigram_vocab_size=int(config.vocab_size),
                    seed=int(getattr(config, "seed", 1234)),
                    ple_dense_layer_id=dense_id,
                )
                sizes, offsets, rows = type(layer)._make_vocab_layout(
                    ngram_vocab_size_base=int(config.ngram_vocab_size_base),
                    ngram_heads=count,
                    ple_dense_layer_id=dense_id,
                )
                divisor = int(config.make_ngram_vocab_size_divisible_by)
                rows = ((rows + divisor - 1) // divisor) * divisor
                saved = getattr(layer, "_inprocess_hash_weights", {})
                values = []
                for key, value in [
                    ("layer_multipliers", multipliers),
                    ("ngram_heads_vocab_sizes", sizes),
                    ("ngram_heads_offsets", offsets),
                ]:
                    tensor = (
                        saved.get(
                            key, torch.tensor(value, dtype=torch.int64, device="cpu")
                        )
                        .to(device="cpu", dtype=torch.int64)
                        .contiguous()
                    )
                    if tensor.ndim != 1 or tensor.numel() != len(value):
                        raise ValueError(f"Invalid PLE checkpoint hash buffer: {key}")
                    values.append(tensor)
                if (
                    torch.any(values[1] <= 0)
                    or torch.any(values[2] < 0)
                    or torch.any(values[1] > rows - values[2])
                ):
                    raise ValueError("PLE checkpoint layout exceeds table")
                path = Path(os.environ["VLLM_PLE_PACKED_TABLE_DIR"]) / (
                    name + ".ngram_embedding.packed_u8"
                )
                meta = json.loads(Path(str(path) + ".json").read_text())
                if (
                    meta.get("row_format") != "fp8_e4m3"
                    or int(meta["total_rows"]) != rows
                    or int(meta["row_width"]) != width
                ):
                    raise ValueError("Packed PLE metadata does not match model")
                reader = BatchReader(
                    path,
                    rows,
                    width,
                    cfg.max_num_batched_tokens * count,
                    cache_mb * 2**20,
                )
                self._readers.append(reader)
                output = torch.empty(
                    (cfg.max_num_batched_tokens, count * width),
                    dtype=dtype,
                    device="cpu",
                ).share_memory_()
                self._owners.append(output)
                sem = CpuGpuSemaphore(torch.device("cpu"))
                sem._flag_tensor = mapped_flag
                layer.setup_cross_process_offload(
                    map_shared_tensor(output, device), sem
                )
                ids = torch.empty(
                    (cfg.max_num_batched_tokens, count), dtype=torch.int64, device="cpu"
                )
                self._hash_state.append(
                    (ngram, heads, int(config.eos_token_id), values, ids)
                )
        except BaseException:
            self.close()
            raise

    def prepare_forward(self, num_reqs, num_tokens, dummy_run):
        if self._closed:
            raise RuntimeError("PLE connector is closed")
        if dummy_run:
            self.signal_dummy_outputs(num_tokens)
            return
        if (
            not 0 < num_reqs <= self._ngram_context_buf.shape[0]
            or not num_reqs <= num_tokens <= self._input_ids_buf.numel()
        ):
            raise ValueError("PLE batch exceeds configured bounds")
        if self._gb10_event_recorded:
            self._gb10_consumed_event.synchronize()
        if self._uses_cuda_inputs:
            self._input_ready_event.record(torch.cuda.current_stream(self.device))
            self._copy_cuda_inputs(
                PleOffloadRequest(
                    dp_rank=0,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    seq=self._seq + 1,
                )
            )
        else:
            self._copy_cpu_inputs(
                PleOffloadRequest(
                    dp_rank=0,
                    num_tokens=num_tokens,
                    num_reqs=num_reqs,
                    seq=self._seq + 1,
                )
            )
        tokens = self._input_ids_buf[:num_tokens].to(torch.int64).clamp_min_(0)
        query = self._query_start_loc_buf[: num_reqs + 1].to(torch.int64)
        history = self._ngram_context_buf[:num_reqs].to(torch.int64).contiguous()
        self._seq += 1
        publish_expected(self._done_flag, self._seq)
        for reader, output, state in zip(self._readers, self._owners, self._hash_state):
            ngram, heads, eos, values, ids = state
            code = self._hash(
                tokens.data_ptr(),
                num_tokens,
                query.data_ptr(),
                num_reqs,
                history.data_ptr(),
                history.stride(0),
                ngram,
                heads,
                eos,
                *(v.data_ptr() for v in values),
                ids.data_ptr(),
            )
            if code:
                raise RuntimeError(f"PLE native hash rejected inputs: {code}")
            reader.gather(ids[:num_tokens], output[:num_tokens], self._timeout_ms)
        publish_done(self._done_flag, self._seq)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._gb10_event_recorded:
            self._gb10_consumed_event.synchronize()
        for reader in self._readers:
            reader.close()
        if self._pinned_input_buffers:
            self._unpin_input_buffers()
