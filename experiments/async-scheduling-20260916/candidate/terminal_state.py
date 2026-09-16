"""Terminal-step selection for versioned async completion snapshots.

Not installed. This identifies only statuses eligible for CompletionCache:
FINISHED_STOPPED and FINISHED_LENGTH_CAPPED. Repetition/abort snapshots remain
ineligible, just as on the synchronous path. Output processors own stop strings.
The device marker must run after sampled-token postprocessing and before a
subsequent forward can overwrite recurrent or circular state.
"""


def terminal_boundary(tokens, *, total_before, prompt_len, max_tokens,
                      max_model_len, eos_token_id, stop_token_ids=()):
    """CPU oracle matching scheduler check_stop's eligible stopping rules."""
    for n, token in enumerate(tokens, 1):
        total = total_before + n
        if (token == eos_token_id or token in stop_token_ids
                or total >= max_model_len or total-prompt_len >= max_tokens):
            return total-1
    return None


def make_marker():
    # Import lazily: CPU ownership and boundary tests need no CUDA initialization.
    import triton
    import triton.language as tl

    @triton.jit
    def mark_terminal(
        idx_mapping, sampled, num_sampled, total_len_after, prompt_len,
        max_tokens, eos_tokens, stop_tokens, stop_counts,
        terminal_boundaries, terminal_steps,
        sampled_stride: tl.constexpr, stop_stride: tl.constexpr,
        max_model_len: tl.constexpr, step,
        STOP_TILE: tl.constexpr,
    ):
        batch_row = tl.program_id(0)
        ri = tl.load(idx_mapping + batch_row)
        if ri < 0:
            return
        # A later in-flight batch must never replace the first terminal version.
        if tl.load(terminal_steps + ri) >= 0:
            return
        n = tl.load(num_sampled + batch_row)
        if n <= 0:
            return
        before = tl.load(total_len_after + ri)-n
        prompt = tl.load(prompt_len + ri)
        limit = tl.load(max_tokens + ri)
        eos = tl.load(eos_tokens + ri)
        offsets = tl.arange(0, STOP_TILE)
        count = tl.load(stop_counts + ri)
        stops = tl.load(stop_tokens + ri*stop_stride + offsets,
                        mask=offsets < count, other=-1)
        boundary = -1
        for i in range(n):
            token = tl.load(sampled + batch_row*sampled_stride + i)
            total = before+i+1
            stopped = ((token == eos) | (tl.sum(((offsets < count) & (stops == token)).to(tl.int32), 0) > 0)
                       | (total >= max_model_len) | (total-prompt >= limit))
            if (boundary < 0) & stopped:
                boundary = total-1
        if boundary >= 0:
            tl.store(terminal_boundaries + ri, boundary)
            tl.store(terminal_steps + ri, step)

    return mark_terminal


def make_snapshot_copier():
    import triton
    import triton.language as tl

    @triton.jit
    def capture_terminal(
        descriptors, computed, accepted, columns, boundaries, terminal_steps,
        step, MAX_SPEC: tl.constexpr, TILE: tl.constexpr,
        COPY_LANES: tl.constexpr,
    ):
        # One descriptor per meaningful tensor piece, not padded KV page.
        # Fixed grid per piece: nonterminal requests exit before the byte loop.
        # No CPU token read and no copy of continuing requests' recurrent state.
        piece = tl.program_id(1)
        lane = tl.program_id(0)
        d = descriptors + piece*15
        ri = tl.load(d+7)
        if tl.load(terminal_steps+ri) != step:
            return
        src = tl.load(d).to(tl.pointer_type(tl.uint8))
        dst = tl.load(d+1).to(tl.pointer_type(tl.uint8))
        length = tl.load(d+2)
        stride = tl.load(d+3)
        block_count = tl.load(d+4)
        table = tl.load(d+5).to(tl.pointer_type(tl.int32))
        table_length = tl.load(d+6)
        axis_stride = tl.load(d+8)
        axis_length = tl.load(d+9)
        circular = tl.load(d+10) != 0
        status = tl.load(d+11).to(tl.pointer_type(tl.int32))
        tag = tl.load(d+12).to(tl.pointer_type(tl.int64))
        writes_tag = tl.load(d+13)
        # d+14 is the immutable CPU-assigned request generation/lease identity.
        generation = tl.load(d+14)
        boundary = tl.load(boundaries+ri)
        end = tl.load(computed+ri)
        bias = tl.load(accepted+ri)-1-(end-boundary)
        column = tl.load(columns+ri)
        selected = tl.where(circular, 0, column+tl.where(axis_stride == 0, bias, 0))
        valid = ((boundary > 0) & (selected >= 0) & (selected < table_length)
                 & (circular | ((end >= boundary) & (bias >= 0)
                                & (bias <= MAX_SPEC) & (column >= 0))))
        block = tl.load(table+selected, mask=valid, other=0)
        valid = valid & (block > 0) & (block < block_count)
        if lane == 0:
            tl.store(status, valid.to(tl.int32))
            if writes_tag:
                tl.store(tag, boundary)
                tl.store(tag+1, step)
                tl.store(tag+2, generation)
        if not valid:
            return
        effective_bias = tl.where(circular, 0, bias)
        for start in range(lane*TILE, length, TILE*COPY_LANES):
            offset = start+tl.arange(0, TILE)
            time = (offset//tl.maximum(axis_stride, 1)) % tl.maximum(axis_length, 1)
            keep = circular | (axis_stride == 0) | (time+effective_bias < axis_length)
            value = tl.load(src+block*stride+offset+effective_bias*axis_stride,
                            mask=(offset < length) & keep, other=0)
            tl.store(dst+offset, value, mask=offset < length)

    return capture_terminal
