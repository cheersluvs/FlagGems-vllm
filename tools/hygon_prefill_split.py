"""Decode's two biggest levers, on prefill: split the row, launch directly.

Neither has ever been tried here. The shipped prefill override only picks a
launch geometry and routes dense shapes to a prefix-sum slot allocator; the
work itself is still one generic launch with grid = num_rows. The note in
memory saying a prefill row split was refuted came from a WALL-time ceiling
test -- the same method that was withdrawn for decode, where the split turned
out to be worth 3.65x at one row once the host dispatch was taken out of the
measurement.

And the three shapes prefill loses worst on are exactly the grid-starved ones:

    (64,129280) k=1024   64 programs on 80 SMs   0.463
    (4,16385)   k=512     4 programs             0.551
    (4,8193)    k=512     4 programs             0.606

while the four many-row shapes already have 4100-16383 programs and read at
their best with no split at all. Supporting evidence from the bandwidth probe:
on (64,129280) the histogram pass goes 167 -> 301 GB/s from split 1 to split 4.

The pipeline, all launches direct:

    bounds   per-chunk [start, end) as ABSOLUTE offsets
    stage1   the generic prefill kernel over every chunk of every row. Prefill
             rows are padded (stride0 > vocab), so decode's trick of passing
             stride0 = CHUNK does not work; instead the row offset is folded
             into the bounds and stride0 is passed as 0, which leaves the
             kernel's own float4 alignment arithmetic correct.
    gather   chunk-local indices -> candidate values and row-relative indices
    tail     the decode override's kernel, reused: the candidate count is
             always split * top_k, so its fallback branch never fires and only
             the exact radix half runs

Reported in BOTH device and wall time: the split trades one launch for four,
and at four rows the shape is small enough that launches may decide it.

    tools/vendor_probe.sh tools/hygon_prefill_split.py hygon_prefill_split
"""

import sys
from importlib import import_module

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

import flaggems_vllm

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")

SHAPES = [
    (64, 129280, 1024, 129280, 1),
    (4, 8193, 512, 8456, 1),
    (4, 16385, 512, 16648, 1),
    (16383, 4095, 512, 4352, 1),
    (12961, 4100, 512, 4360, 1),
    (16380, 5115, 512, 5376, 1),
    (4100, 1025, 512, 1288, 1),
]
SPLITS = (1, 2, 4, 8, 16, 32)
MIN_CHUNK = 1024
RADIX = 256


def wall_us(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000


def device_us(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total = 0.0
    for ev in prof.key_averages():
        t = getattr(ev, "self_device_time_total", None)
        if t is None:
            t = getattr(ev, "self_cuda_time_total", 0)
        total += t or 0.0
    return total / iters


@triton.jit
def _bounds(
    starts_ptr,
    ends_ptr,
    out_start_ptr,
    out_end_ptr,
    stride0,
    SPLIT: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Absolute [start, end) per chunk, so stage 1 can be told stride0 = 0."""
    row = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    m = c < SPLIT
    base = row * stride0 + tl.load(starts_ptr + row)
    end = row * stride0 + tl.load(ends_ptr + row)
    s = base + c * CHUNK
    tl.store(out_start_ptr + row * SPLIT + c, tl.minimum(s, end).to(tl.int32), mask=m)
    tl.store(
        out_end_ptr + row * SPLIT + c,
        tl.minimum(s + CHUNK, end).to(tl.int32),
        mask=m,
    )


@triton.jit
def _gather(
    logits_ptr,
    starts_ptr,
    cand_ptr,
    cand_val_ptr,
    cand_idx_ptr,
    stride0,
    floor,
    SPLIT: tl.constexpr,
    TOPK: tl.constexpr,
    CHUNK: tl.constexpr,
    NCAND: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Slot p of a row's candidate array is chunk p // TOPK, slot p % TOPK.
    A padding index (-1, from a chunk past the row's end) becomes `floor`, and
    the index written is relative to the ROW's start, which is what the
    operator returns."""
    row = tl.program_id(0)
    p = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = p < NCAND
    chunk_id = p // TOPK
    local = tl.load(
        cand_ptr + (row * SPLIT + chunk_id) * TOPK + (p % TOPK), mask=m, other=-1
    )
    ok = m & (local >= 0)
    rel = chunk_id * CHUNK + local
    base = row * stride0 + tl.load(starts_ptr + row)
    val = tl.load(logits_ptr + base + rel, mask=ok, other=floor)
    tl.store(cand_val_ptr + row * NCAND + p, tl.where(ok, val, floor), mask=m)
    tl.store(
        cand_idx_ptr + row * NCAND + p,
        tl.where(ok, rel, -1).to(tl.int32),
        mask=m,
    )


def main():
    gen = _generic
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    import vllm._custom_ops  # noqa: F401

    dev = "cuda"
    sms = ov._sm_count()
    block = gen.NUM_THREADS_PER_BLOCK
    print(f"{sms} SMs; ratio vs vLLM (wall), device us in brackets\n")
    for rows, vocab, top_k, stride0, stride1 in SHAPES:
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)
        want = torch.topk(logits, top_k, dim=1).values.sort(dim=1).values
        floor = torch.finfo(torch.float32).min

        t_vllm = wall_us(
            lambda: torch.ops._C.top_k_per_row_prefill(
                logits, starts, ends, idx, rows, stride0, stride1, top_k
            )
        )
        t_ship = wall_us(
            lambda: flaggems_vllm.top_k_per_row_prefill(
                logits, starts, ends, idx, rows, stride0, stride1, top_k
            )
        )
        d_ship = device_us(
            lambda: flaggems_vllm.top_k_per_row_prefill(
                logits, starts, ends, idx, rows, stride0, stride1, top_k
            )
        )
        print(
            f"  {rows} x {vocab}, top_k {top_k}: vLLM {t_vllm:.1f} us, "
            f"shipped {t_vllm / t_ship:.3f} ({d_ship:.1f} us device)"
        )
        splits = SPLITS if rows < 4 * sms else (1,)
        for split in splits:
            chunk = triton.cdiv(vocab, split)
            if split > 1 and chunk < MIN_CHUNK:
                continue
            nv = rows * split
            ncand = split * top_k
            cap = triton.next_power_of_2(ncand)
            cstart = torch.empty((nv,), dtype=torch.int32, device=dev)
            cend = torch.empty((nv,), dtype=torch.int32, device=dev)
            cand = torch.empty((nv, top_k), dtype=torch.int32, device=dev)
            cand_val = torch.empty((rows, cap), dtype=torch.float32, device=dev)
            cand_idx = torch.empty((rows, cap), dtype=torch.int32, device=dev)
            cnt = torch.full((rows,), ncand, dtype=torch.int32, device=dev)
            hist = torch.empty((rows, 2048), dtype=torch.int32, device=dev)
            counts = torch.empty((rows, RADIX), dtype=torch.int32, device=dev)
            slot = torch.empty((rows,), dtype=torch.int32, device=dev)
            scratch = (
                torch.empty((nv, gen.NUM_BINS), dtype=torch.int32, device=dev),
                torch.empty(
                    (nv, gen.NUM_FILNAL_ITEMS), dtype=torch.float32, device=dev
                ),
                torch.empty((nv,), dtype=torch.int32, device=dev),
                torch.empty((nv,), dtype=torch.int32, device=dev),
                torch.empty((nv,), dtype=torch.int32, device=dev),
                torch.empty((nv,), dtype=torch.int32, device=dev),
            )
            gblock = min(1024, triton.next_power_of_2(ncand))
            lb = ov._Launch(
                _bounds,
                (rows,),
                {
                    "SPLIT": split,
                    "CHUNK": chunk,
                    "BLOCK": triton.next_power_of_2(split),
                },
                1,
            )
            l1 = ov._Launch(
                gen.non_tle_top_k_per_row_prefill,
                (nv,),
                {"TOPK": top_k, "BLOCK_SIZE": block, "ROW_OFFSET": 0},
                gen._num_warps(block),
            )
            lg = ov._Launch(
                _gather,
                (rows, triton.cdiv(ncand, gblock)),
                {
                    "SPLIT": split,
                    "TOPK": top_k,
                    "CHUNK": chunk,
                    "NCAND": ncand,
                    "BLOCK": gblock,
                },
                4,
            )
            lt = ov._Launch(
                ov._tail,
                (rows,),
                {
                    "TOPK": top_k,
                    "NB": 2048,
                    "CAP": cap,
                    "RADIX": RADIX,
                    "BLOCK": 512,
                },
                8,
            )

            def run():
                lb(starts, ends, cstart, cend, stride0)
                l1(logits, cand, cstart, cend, 0, 1, chunk, *scratch)
                lg(logits, starts, cand, cand_val, cand_idx, stride0, floor)
                lt(
                    logits,
                    ends,
                    hist,
                    cnt,
                    cand_idx,
                    cand_val,
                    idx,
                    counts,
                    slot,
                    stride0,
                )

            idx.fill_(-9)
            run()
            torch.cuda.synchronize()
            got = logits.gather(1, idx.long().clamp(0, vocab - 1)).sort(dim=1).values
            ok = torch.allclose(got, want) and bool((idx >= 0).all())
            t = wall_us(run)
            d = device_us(run)
            print(
                f"      split {split:>2}: {t_vllm / t:>6.3f} ({d:>7.1f} us device, "
                f"{nv} programs, {ncand} candidates) "
                f"{'OK' if ok else 'WRONG'}"
            )
        print()


if __name__ == "__main__":
    sys.exit(main())
