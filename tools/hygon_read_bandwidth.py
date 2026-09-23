"""How fast can this card read -- and how fast can one-program-per-row read?

The dense budget (tools/hygon_prefill_dense_budget.py) puts the first read of
each row plus its key math at ~209 us on 12961x4100, 212 MB, about 1.0 TB/s.
Whether that has headroom depends on what this card can actually deliver,
which has only ever been inferred ("the generic collection pass reaches ~1270
GB/s"). This measures it directly with plain streaming kernels -- no override,
no atomics, every load's value folded into a per-program sum that is stored,
so nothing can be optimised away.

PEAK. A 1-D kernel over one contiguous 256 MB buffer, unmasked, a few
geometries. The best of them is the ceiling any row pattern can hope for.

ROW PATTERN, on the three dense shapes with the benchmark's padded stride0,
one program per row at the dense route's geometry (BLOCK 256, 2 warps), bulk
loop unmasked and remainder masked, exactly as the dense kernel reads:

    v2        VEC 2 (what the dense route ships)
    v4        VEC 4, 128-bit loads
    v4nohint  VEC 4 without tl.multiple_of on the row pointer: stride0 is not a
              multiple of 16 (4352/4360/5376 floats), so Triton cannot prove
              alignment by itself -- does the hint matter?
    v2key     v2 plus the STEP-0 key math (fp16 convert and mapping): the dense
              kernel's P1 without its atomics
    v2twice   the row read twice in the same program: the second read's
              marginal cost says whether it is served by L2

GB/s counts row bytes only (num_rows * vocab * 4; twice for v2twice). do_bench,
min of three.

    tools/vendor_probe.sh tools/hygon_read_bandwidth.py hygon_read_bandwidth
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _flat_read(
    x_ptr, out_ptr, BLOCK: tl.constexpr, VEC: tl.constexpr, ITERS: tl.constexpr
):
    pid = tl.program_id(0)
    off = tl.arange(0, BLOCK)[:, None] * VEC + tl.arange(0, VEC)[None, :]
    base = pid * (BLOCK * VEC * ITERS)
    acc = tl.zeros([BLOCK, VEC], tl.float32)
    for t in tl.range(0, ITERS):
        acc += tl.load(x_ptr + base + t * BLOCK * VEC + off)
    tl.store(out_ptr + pid, tl.sum(tl.sum(acc, axis=1), axis=0))


@triton.jit
def _val(x, KEY: tl.constexpr):
    if KEY:
        h = x.to(tl.float16)
        bits = h.to(tl.uint16, bitcast=True)
        sign_set = (bits & 0x8000) != 0
        inv = (~bits) & 0x7FFF
        mapped = tl.where(sign_set, bits, inv)
        return (mapped >> 5).to(tl.int32)
    else:
        return x.to(tl.int32, bitcast=True)


@triton.jit
def _row_read(
    x_ptr,
    out_ptr,
    stride0,
    vocab,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
    KEY: tl.constexpr,
    PASSES: tl.constexpr,
    HINT: tl.constexpr,
):
    row = tl.program_id(0)
    base = x_ptr + row * stride0
    if HINT:
        base = tl.multiple_of(base, VEC * 4)
    lane = tl.arange(0, BLOCK)
    off = lane[:, None] * VEC + tl.arange(0, VEC)[None, :]
    n_full = vocab // (BLOCK * VEC)
    rem = n_full * BLOCK * VEC
    acc = tl.zeros([BLOCK, VEC], tl.int32)
    acc_r = tl.zeros([BLOCK], tl.int32)
    for p in tl.static_range(PASSES):
        for t in tl.range(0, n_full):
            acc += _val(tl.load(base + t * BLOCK * VEC + off), KEY)
        for t in tl.range(0, tl.cdiv(vocab - rem, BLOCK)):
            i = rem + t * BLOCK + lane
            x = tl.load(base + i, mask=i < vocab, other=0.0)
            acc_r += tl.where(i < vocab, _val(x, KEY), 0)
    tl.store(out_ptr + row, tl.sum(tl.sum(acc, axis=1), axis=0) + tl.sum(acc_r, axis=0))


def bench(fn):
    return min(triton.testing.do_bench(fn, warmup=25, rep=300) for _ in range(3)) * 1e3


def main():
    dev = "cuda"
    print("### peak: one contiguous 256 MB buffer, unmasked\n")
    n = 64 * 1024 * 1024
    x = torch.randn(n, device=dev, dtype=torch.float32)
    best = 0.0
    for block, vec, warps, iters in (
        (256, 4, 4, 8),
        (512, 4, 8, 8),
        (1024, 4, 8, 4),
        (256, 4, 4, 32),
        (512, 2, 4, 16),
    ):
        per = block * vec * iters
        grid = n // per
        out = torch.empty(grid, device=dev, dtype=torch.float32)
        us = bench(
            lambda: _flat_read[(grid,)](
                x, out, BLOCK=block, VEC=vec, ITERS=iters, num_warps=warps
            )
        )
        gbs = grid * per * 4 / us / 1e3
        best = max(best, gbs)
        print(
            f"  BLOCK {block:4d} VEC {vec} warps {warps} ITERS {iters:2d}"
            f"   {us:8.1f} us   {gbs:7.0f} GB/s"
        )
    print(f"\n  peak read: {best:.0f} GB/s")
    x = None  # release before the row buffers
    torch.cuda.empty_cache()

    arms = [
        ("v2", 2, False, 1, True),
        ("v4", 4, False, 1, True),
        ("v4nohint", 4, False, 1, False),
        ("v2key", 2, True, 1, True),
        ("v2twice", 2, False, 2, True),
    ]
    print("\n### row pattern: one program per row, BLOCK 256, 2 warps\n")
    print(f"  {'shape':>12} {'arm':>9} {'us':>9} {'GB/s':>7} {'of peak':>8}")
    for num_rows, vocab, stride0 in (
        (16383, 4095, 4352),
        (12961, 4100, 4360),
        (16380, 5115, 5376),
    ):
        torch.manual_seed(42)
        buf = torch.randn(
            (num_rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32
        )
        out = torch.empty(num_rows, device=dev, dtype=torch.int32)
        once = None
        for tag, vec, key, passes, hint in arms:
            us = bench(
                lambda: _row_read[(num_rows,)](
                    buf,
                    out,
                    stride0,
                    vocab,
                    BLOCK=256,
                    VEC=vec,
                    KEY=key,
                    PASSES=passes,
                    HINT=hint,
                    num_warps=2,
                )
            )
            nbytes = num_rows * vocab * 4 * passes
            gbs = nbytes / us / 1e3
            note = ""
            if tag == "v2":
                once = us
            if tag == "v2twice" and once:
                note = f"   second read adds {us - once:.1f} us"
            print(
                f"  {num_rows}x{vocab:<5} {tag:>9} {us:9.1f} {gbs:7.0f} {100 * gbs / best:7.0f}%{note}"
            )
        buf = None
        torch.cuda.empty_cache()

    print(
        "\n  For reference, the dense kernel's first read + key math (budget,"
        " cut1r - cut0): 249 / 209 / 313 us on the three shapes."
    )


if __name__ == "__main__":
    main()
