"""What does the operator pay per program before it touches any data?

The Hygon ablation says the 2048-counter CLEAR costs 11.6 us of the 18.07 us
a prefill program takes, and the single-address slot atomic another 7.55. The
clear cannot be bandwidth: 2048 int32 is 8 KB, four steps is 32 KB per program,
about 0.03 us at this card's bandwidth -- 400x less than measured. So either
the number is real and something else in it dominates (barriers, the store
pattern, residency), or the ablation's delta is contaminated by the control
flow it also removed.

This prices the pieces directly, each as its own kernel at the operator's
geometry (BLOCK=512 on 8 warps, one counter row per program, 40 waves), so the
numbers are independent of that ablation:

    noop            the harness floor
    barriers        8 x tl.debug_barrier() and nothing else
    clear_tiles     the operator's form: 4 tiles of 512 + barrier, one step
    clear_steps     the same, four steps (what the operator actually does)
    clear_wide      one 2048-wide store per step, four steps
    clear_smem      the same four steps in shared memory (TLE), if available
    scan_global     load 2048, cumsum, store back -- one step
    scan_smem       the same in shared memory, if available

Deltas against `noop` are the per-program cost of each piece. If clear_steps
is nowhere near 11.6 us, the ablation delta was control flow, not the clear.

    tools/vendor_probe.sh tools/topk_fixed_cost.py <name>
"""

import sys

import torch
import triton
import triton.language as tl

SMS = int(getattr(torch.cuda.get_device_properties(0), "multi_processor_count", 80))
ROWS = 40 * SMS
NB = 2048
BLOCK = 512
WARPS = 8
STEPS = 4

try:
    from triton._C import libtriton

    HAS_TLE = hasattr(libtriton.ir.builder, "make_swizzled_shared_encoding_attr")
    if HAS_TLE:
        import triton.experimental.tle.language as tle
except Exception:  # noqa: BLE001
    HAS_TLE = False


@triton.jit
def k_noop(hist_ptr, out_ptr, NB: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    tl.store(out_ptr + row, tl.load(hist_ptr + row * NB))


@triton.jit
def k_barriers(
    hist_ptr, out_ptr, NB: tl.constexpr, BLOCK: tl.constexpr, N: tl.constexpr
):
    row = tl.program_id(0)
    for _ in tl.static_range(N):
        tl.debug_barrier()
    tl.store(out_ptr + row, tl.load(hist_ptr + row * NB))


@triton.jit
def k_clear_tiles(
    hist_ptr, out_ptr, NB: tl.constexpr, BLOCK: tl.constexpr, STEPS: tl.constexpr
):
    """The operator's own shape: NB/BLOCK tiles of zeros, then a barrier."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = hist_ptr + row * NB
    for _ in tl.static_range(STEPS):
        for t in tl.static_range(NB // BLOCK):
            tl.store(base + t * BLOCK + lane, tl.zeros([BLOCK], tl.int32))
        tl.debug_barrier()
    tl.store(out_ptr + row, tl.load(base))


@triton.jit
def k_clear_wide(
    hist_ptr, out_ptr, NB: tl.constexpr, BLOCK: tl.constexpr, STEPS: tl.constexpr
):
    """One NB-wide store per step instead of NB/BLOCK narrow ones."""
    row = tl.program_id(0)
    bins = tl.arange(0, NB)
    base = hist_ptr + row * NB
    for _ in tl.static_range(STEPS):
        tl.store(base + bins, tl.zeros([NB], tl.int32))
        tl.debug_barrier()
    tl.store(out_ptr + row, tl.load(base))


@triton.jit
def k_scan_global(hist_ptr, out_ptr, NB: tl.constexpr, BLOCK: tl.constexpr):
    """Read the counters back, prefix-sum them, write them back: one step."""
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    base = hist_ptr + row * NB
    carry = tl.zeros([], tl.int32)
    for t in tl.static_range(NB // BLOCK):
        c = tl.load(base + t * BLOCK + lane)
        pre = tl.cumsum(c, axis=0) - c + carry
        tl.store(base + t * BLOCK + lane, pre)
        carry += tl.sum(c, axis=0)
    tl.debug_barrier()
    tl.store(out_ptr + row, carry)


if HAS_TLE:

    @triton.jit
    def k_clear_smem(
        hist_ptr, out_ptr, NB: tl.constexpr, BLOCK: tl.constexpr, STEPS: tl.constexpr
    ):
        row = tl.program_id(0)
        buf = tle.gpu.alloc(
            [NB],
            dtype=tl.int32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        view = tle.gpu.local_ptr(buf)
        for _ in tl.static_range(STEPS):
            tl.store(view, tl.zeros([NB], tl.int32))
            tl.debug_barrier()
        tl.store(out_ptr + row, tl.sum(tl.load(view), axis=0))

    @triton.jit
    def k_scan_smem(hist_ptr, out_ptr, NB: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        buf = tle.gpu.alloc(
            [NB],
            dtype=tl.int32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        view = tle.gpu.local_ptr(buf)
        tl.store(view, tl.load(hist_ptr + row * NB + tl.arange(0, NB)))
        tl.debug_barrier()
        p = tle.gpu.local_ptr(buf, (0,)) + (tl.program_id(0) >> 31)
        carry = tl.zeros([], tl.int32)
        for t in tl.static_range(NB // BLOCK):
            c = tl.load(p + t * BLOCK + lane)
            pre = tl.cumsum(c, axis=0) - c + carry
            tl.store(p + t * BLOCK + lane, pre)
            carry += tl.sum(c, axis=0)
        tl.debug_barrier()
        tl.store(out_ptr + row, carry)


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000 / (ROWS / SMS)  # us per program


def main():
    dev = "cuda"
    hist = torch.randint(0, 8, (ROWS * NB,), dtype=torch.int32, device=dev)
    out = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    kw = dict(NB=NB, BLOCK=BLOCK, num_warps=WARPS)
    print(
        f"{ROWS} programs = 40 waves on {SMS} SMs | BLOCK={BLOCK} x {WARPS} warps | "
        f"{NB} counters per program | microseconds per program\n"
    )

    base = timed(lambda: k_noop[(ROWS,)](hist, out, **kw))
    rows = [("noop", base)]
    rows.append(
        ("barriers x8", timed(lambda: k_barriers[(ROWS,)](hist, out, N=8, **kw)))
    )
    rows.append(
        (
            "clear_tiles x1 step",
            timed(lambda: k_clear_tiles[(ROWS,)](hist, out, STEPS=1, **kw)),
        )
    )
    rows.append(
        (
            f"clear_tiles x{STEPS} steps",
            timed(lambda: k_clear_tiles[(ROWS,)](hist, out, STEPS=STEPS, **kw)),
        )
    )
    rows.append(
        (
            f"clear_wide x{STEPS} steps",
            timed(lambda: k_clear_wide[(ROWS,)](hist, out, STEPS=STEPS, **kw)),
        )
    )
    rows.append(
        ("scan_global x1 step", timed(lambda: k_scan_global[(ROWS,)](hist, out, **kw)))
    )
    if HAS_TLE:
        rows.append(
            (
                f"clear_smem x{STEPS} steps",
                timed(lambda: k_clear_smem[(ROWS,)](hist, out, STEPS=STEPS, **kw)),
            )
        )
        rows.append(
            ("scan_smem x1 step", timed(lambda: k_scan_smem[(ROWS,)](hist, out, **kw)))
        )
    else:
        print("  (no TLE bindings: shared-memory rows skipped)\n")

    print(f"  {'piece':<26} {'us/prog':>9} {'minus noop':>11}")
    for name, t in rows:
        print(f"  {name:<26} {t:>9.3f} {t - base:>11.3f}")
    print("\n  The ablation attributed 11.6 us/program to the clear and 7.55 to the")
    print("  slot atomic, out of 18.07 for the whole operator. Compare the")
    print(f"  clear_tiles x{STEPS} row against that 11.6 before believing either.")


if __name__ == "__main__":
    sys.exit(main())
