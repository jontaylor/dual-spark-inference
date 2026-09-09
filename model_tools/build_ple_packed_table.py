#!/usr/bin/env python3
"""Build a packed NVFP4 or FP8 PLE table for memory-mapped CPU offload.

The checkpoint stores the PLE n-gram table as 128 row shards, each split into
4-bit codes (uint8 [rows, head_dim/2]) and FP8 block scales ([rows, head_dim/16]).
vLLM's NVFP4 PLE lookup returns, per row, ``cat(codes, scales.view(uint8))``.
This script writes exactly that layout as one flat file: [total_rows, 90] uint8,
row r of shard i at index i*shard_rows + r. The offload worker memory-maps it,
so the 26.8 GiB table lives in the (evictable) page cache instead of RSS.

FP8 checkpoints instead store F8_E4M3 rows. Copy their raw bytes unchanged;
the global scale remains in the checkpoint and is loaded/dequantised by the
existing PLE path. Both formats use .packed_u8 with an explicit row_format
in the JSON sidecar. No quantisation or dequantisation occurs in this builder.

Streams shard-by-shard with numpy memmaps; peak copy buffers are below 1 GiB.

Usage: build_ple_packed_table.py <snapshot_dir> <out_dir>
"""
import json, os, struct, sys, time
import numpy as np

snap, out_dir = sys.argv[1], sys.argv[2]
idx = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
prefix_of = {}
for k in idx:
    if ".ngram_embedding.shard_0.weight" in k and not k.endswith("weight_scale"):
        prefix_of[k[: k.index(".shard_0.weight")]] = True
if not prefix_of:
    sys.exit("no PLE ngram_embedding shards in index")

headers = {}
def header(fname):
    if fname not in headers:
        with open(os.path.join(snap, fname), "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            headers[fname] = (json.loads(f.read(n)), 8 + n)
    return headers[fname]

def view(name):
    fname = idx[name]
    h, base = header(fname)
    meta = h[name]
    start, end = meta["data_offsets"]
    mm = np.memmap(os.path.join(snap, fname), dtype=np.uint8, mode="r",
                   offset=base + start, shape=(end - start,))
    return mm.reshape(meta["shape"]), meta["dtype"]

os.makedirs(out_dir, exist_ok=True)
for prefix in prefix_of:
    # vLLM name: strip leading "model." and map language_model -> language_model.model
    vname = prefix
    if vname.startswith("model.language_model."):
        vname = "language_model.model." + vname[len("model.language_model."):]
    shards = sorted({int(k[len(prefix) + len(".shard_"):].split(".")[0])
                     for k in idx if k.startswith(prefix + ".shard_")})
    assert shards == list(range(len(shards))), shards
    w0, dt = view(f"{prefix}.shard_0.weight")
    if dt == "U8":
        row_format = "nvfp4"
        s0, scale_dt = view(f"{prefix}.shard_0.weight_scale")
        assert scale_dt == "F8_E4M3", scale_dt
        sw = s0.shape[1]
    elif dt == "F8_E4M3":
        row_format = "fp8_e4m3"
        # Keep the original global scale outside the packed row data.
        assert f"{prefix}.weight_scale" in idx, "FP8 PLE global scale missing"
        sw = 0
    else:
        raise ValueError(f"Unsupported PLE storage dtype: {dt}")
    rows, cw = w0.shape
    width = cw + sw
    out_name = os.path.join(out_dir, vname + ".packed_u8")
    meta = {"row_format": row_format, "storage_dtype": dt,
            "rows_per_shard": rows, "num_shards": len(shards), "row_width": width,
            "codes_width": cw, "scales_width": sw, "total_rows": rows * len(shards),
            "snapshot": os.path.basename(os.path.normpath(snap))}
    if os.path.exists(out_name) and os.path.exists(out_name + ".json"):
        old_meta = json.load(open(out_name + ".json"))
        if old_meta == meta and os.path.getsize(out_name) == rows * len(shards) * width:
            print("exists (matching format and revision):", out_name); continue
    print(f"building {out_name}: {len(shards)} shards x {rows} rows x {width} B = "
          f"{rows*len(shards)*width/2**30:.2f} GiB", flush=True)
    t0 = time.time()
    tmp = out_name + ".tmp"
    CH = 1 << 19
    with open(tmp, "wb") as out:
        for i in shards:
            w, shard_dt = view(f"{prefix}.shard_{i}.weight")
            assert w.shape == (rows, cw) and shard_dt == dt, (i, w.shape, shard_dt)
            if row_format == "nvfp4":
                s, scale_dt = view(f"{prefix}.shard_{i}.weight_scale")
                assert s.shape == (rows, sw) and scale_dt == "F8_E4M3", (i, s.shape)
            for c in range(0, rows, CH):
                if row_format == "nvfp4":
                    np.concatenate([w[c:c + CH], s[c:c + CH]], axis=1).tofile(out)
                else:
                    w[c:c + CH].tofile(out)
            if i % 8 == 0:
                print(f"  shard {i}/{len(shards)} {time.time()-t0:.0f}s", flush=True)
        out.flush()
        os.fsync(out.fileno())
    assert os.path.getsize(tmp) == rows * len(shards) * width
    os.replace(tmp, out_name)
    with open(out_name + ".json.tmp", "w") as f:
        json.dump(meta, f, indent=1)
        f.flush(); os.fsync(f.fileno())
    os.replace(out_name + ".json.tmp", out_name + ".json")
    print(f"done in {time.time()-t0:.0f}s", flush=True)
