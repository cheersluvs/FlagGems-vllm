"""One DSA/UB experiment per process, named by argv[1].

Probe 2 could not tell failure from contamination: the first kernel timed out
the vector core, and every later launch reported "Failed to submit kernel task"
-- the same sticky-error cascade seen on MUSA.  So each case runs alone here,
and the driver gives each its own process and log.

The cases answer, in order: does allocation alone hang; can a UB buffer be read
as a tensor; can data be staged GM->UB->GM; is there any pointer to a UB buffer;
and can a histogram scatter (atomic_add at a computed index) reach UB at all.
That last one decides whether the operator's radix histogram can live on chip,
because DSA exposes copy/to_tensor/insert_slice and no scatter primitive.
"""

import os
import sys
import traceback

import torch
import triton
import triton.language as tl

try:
    import torch_npu  # noqa: F401
except Exception:
    pass

import triton.experimental.tle.language.dsa as dsa

CASE = sys.argv[1]
B = 128
UB = dsa.ascend.UB


@triton.jit
def k_alloc_only(out_ptr, BLOCK: tl.constexpr):
    dsa.alloc([BLOCK], tl.int32, dsa.ascend.UB)
    lane = tl.arange(0, BLOCK)
    tl.store(out_ptr + lane, lane)


@triton.jit
def k_to_tensor(out_ptr, BLOCK: tl.constexpr):
    buf = dsa.alloc([BLOCK], tl.int32, dsa.ascend.UB)
    tl.store(out_ptr + tl.arange(0, BLOCK), dsa.to_tensor(buf))


@triton.jit
def k_copy(in_ptr, out_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + lane)
    buf = dsa.alloc([BLOCK], tl.int32, dsa.ascend.UB)
    dsa.copy(dsa.to_buffer(x), buf, [BLOCK])
    tl.store(out_ptr + lane, dsa.to_tensor(buf) * 2)


@triton.jit
def k_ptr_store(out_ptr, BLOCK: tl.constexpr):
    buf = dsa.alloc([BLOCK], tl.int32, dsa.ascend.UB)
    p = dsa.core.from_buffer_to_tensor_pointer(buf)
    lane = tl.arange(0, BLOCK)
    tl.store(p + lane, lane * 3)
    tl.debug_barrier()
    tl.store(out_ptr + lane, tl.load(p + lane))


@triton.jit
def k_atomic(idx_ptr, out_ptr, BLOCK: tl.constexpr, NBIN: tl.constexpr):
    buf = dsa.alloc([NBIN], tl.int32, dsa.ascend.UB)
    p = dsa.core.from_buffer_to_tensor_pointer(buf)
    lane = tl.arange(0, BLOCK)
    idx = tl.load(idx_ptr + lane)
    tl.atomic_add(p + idx, 1)
    tl.debug_barrier()
    tl.store(out_ptr + tl.arange(0, NBIN), tl.load(p + tl.arange(0, NBIN)))


@triton.jit
def k_cap(out_ptr, N: tl.constexpr):
    buf = dsa.alloc([N], tl.int32, dsa.ascend.UB)
    tl.store(out_ptr + tl.arange(0, N), dsa.to_tensor(buf))


@triton.jit
def k_atomic2(idx_ptr, out_ptr, N: tl.constexpr):
    """Close the pointer question the shape mismatch left open."""
    buf = dsa.alloc([N], tl.int32, dsa.ascend.UB)
    p = dsa.core.from_buffer_to_tensor_pointer(buf)
    lane = tl.arange(0, N)
    tl.atomic_add(p + tl.load(idx_ptr + lane), 1)
    tl.store(out_ptr + lane, tl.load(p + lane))


@triton.jit
def k_cap_copy(in_ptr, out_ptr, N: tl.constexpr):
    """Capacity measured on the idiom that works, not on the one that hangs."""
    lane = tl.arange(0, N)
    x = tl.load(in_ptr + lane)
    buf = dsa.alloc([N], tl.int32, dsa.ascend.UB)
    dsa.copy(dsa.to_buffer(x), buf, [N])
    tl.store(out_ptr + lane, dsa.to_tensor(buf))


@triton.jit
def k_hist(in_ptr, out_ptr, BLOCK: tl.constexpr, NBINS: tl.constexpr):
    """A scatter-free histogram: no UB pointer needed, no per-element atomic."""
    x = tl.load(in_ptr + tl.arange(0, BLOCK))
    tl.store(out_ptr + tl.arange(0, NBINS), tl.histogram(x, NBINS))


@triton.jit
def k_hist_accum(in_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr,
                 NBINS: tl.constexpr):
    """The shape the operator would use: accumulate a row, flush bins once."""
    acc = tl.zeros([NBINS], tl.int32)
    for off in tl.range(0, N, BLOCK):
        x = tl.load(in_ptr + off + tl.arange(0, BLOCK))
        acc += tl.histogram(x, NBINS)
    tl.store(out_ptr + tl.arange(0, NBINS), acc)


# --- timing: is tl.histogram actually faster than the per-element atomic? ---
# Correct is not fast.  The histogram pass is where this operator loses ~100x,
# so the only number that matters is how these two lower against each other.
# N is a power of two near the real 129280 vocab so the tile loop divides.

@triton.jit
def k_time_atomic(in_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr,
                  NBINS: tl.constexpr):
    row = tl.program_id(0)
    base = in_ptr + row * N
    hp = out_ptr + row * NBINS
    for off in tl.range(0, N, BLOCK):
        b = tl.load(base + off + tl.arange(0, BLOCK))
        tl.atomic_add(hp + b, 1)


@triton.jit
def k_time_hist(in_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr,
                NBINS: tl.constexpr):
    row = tl.program_id(0)
    base = in_ptr + row * N
    acc = tl.zeros([NBINS], tl.int32)
    for off in tl.range(0, N, BLOCK):
        acc += tl.histogram(tl.load(base + off + tl.arange(0, BLOCK)), NBINS)
    tl.store(out_ptr + row * NBINS + tl.arange(0, NBINS), acc)


@triton.jit
def k_time_read(in_ptr, out_ptr, N: tl.constexpr, BLOCK: tl.constexpr,
                NBINS: tl.constexpr):
    """Control: same traffic, no binning.  Without it a ratio means nothing."""
    row = tl.program_id(0)
    base = in_ptr + row * N
    s = tl.zeros([BLOCK], tl.int32)
    for off in tl.range(0, N, BLOCK):
        s += tl.load(base + off + tl.arange(0, BLOCK))
    tl.store(out_ptr + row * NBINS + tl.arange(0, BLOCK), s)


def run():
    if CASE.startswith("time_"):
        import time
        # time_hist_b4096 -> BLOCK 4096.  The tile width is the knob that
        # matters: each tile pays one NBINS-wide accumulate regardless of how
        # many elements it carries, so at BLOCK 512 the accumulator does 4x
        # more adds than there are elements to bin.
        ROWS, N, NB, BLK = 64, 131072, 2048, 512
        if "_b" in CASE:
            BLK = int(CASE.split("_b")[-1])
        x = torch.randint(0, NB, (ROWS, N), dtype=torch.int32, device="npu")
        out = torch.zeros(ROWS, NB, dtype=torch.int32, device="npu")
        stem = CASE.split("_b")[0]
        fn = {"time_atomic": k_time_atomic, "time_hist": k_time_hist,
              "time_read": k_time_read}[stem]

        def once():
            out.zero_()
            fn[(ROWS,)](x, out, N=N, BLOCK=BLK, NBINS=NB)

        for _ in range(3):
            once()
        torch.npu.synchronize()
        ts = []
        for _ in range(10):
            t0 = time.perf_counter()
            once()
            torch.npu.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        ts.sort()
        chk = ""
        if stem in ("time_atomic", "time_hist"):
            ref = torch.zeros(ROWS, NB, dtype=torch.int32)
            for r in range(ROWS):
                ref[r] = torch.bincount(x[r].cpu().long(), minlength=NB)
            chk = " | " + ("CORRECT" if torch.equal(out.cpu(), ref) else "WRONG")
        return (f"{ROWS}x{N} into {NB} bins, BLOCK {BLK} | min {ts[0]:.2f} med "
                f"{ts[len(ts) // 2]:.2f} max {ts[-1]:.2f} ms{chk}")

    if CASE == "atomic2":
        N = 128
        idx = torch.randint(0, N, (N,), dtype=torch.int32, device="npu")
        out = torch.zeros(N, dtype=torch.int32, device="npu")
        k_atomic2[(1,)](idx, out, N=N)
        torch.npu.synchronize()
        ref = torch.bincount(idx.cpu().long(), minlength=N).to(torch.int32)
        return f"ran, {'CORRECT' if torch.equal(out.cpu(), ref) else 'WRONG'}"

    if CASE.startswith("capcopy"):
        n = int(CASE[7:])
        src_t = torch.arange(n, dtype=torch.int32, device="npu")
        out = torch.zeros(n, dtype=torch.int32, device="npu")
        k_cap_copy[(1,)](src_t, out, N=n)
        torch.npu.synchronize()
        ok = torch.equal(out, src_t)
        return f"N={n} ({n * 4 // 1024} KB) ran, {'CORRECT' if ok else 'WRONG'}"

    if CASE == "hist":
        NB, BLK = 2048, 512
        x = torch.randint(0, NB, (BLK,), dtype=torch.int32, device="npu")
        out = torch.zeros(NB, dtype=torch.int32, device="npu")
        k_hist[(1,)](x, out, BLOCK=BLK, NBINS=NB)
        torch.npu.synchronize()
        ref = torch.bincount(x.cpu().long(), minlength=NB).to(torch.int32)
        return (f"ran, {'CORRECT' if torch.equal(out.cpu(), ref) else 'WRONG'} "
                f"| sum={int(out.sum())} expected {BLK}")

    if CASE == "hist_accum":
        NB, BLK, N = 2048, 512, 512 * 32
        x = torch.randint(0, NB, (N,), dtype=torch.int32, device="npu")
        out = torch.zeros(NB, dtype=torch.int32, device="npu")
        k_hist_accum[(1,)](x, out, N=N, BLOCK=BLK, NBINS=NB)
        torch.npu.synchronize()
        ref = torch.bincount(x.cpu().long(), minlength=NB).to(torch.int32)
        return (f"ran, {'CORRECT' if torch.equal(out.cpu(), ref) else 'WRONG'} "
                f"| sum={int(out.sum())} expected {N}")

    if CASE == "alloc_only":
        out = torch.zeros(B, dtype=torch.int32, device="npu")
        k_alloc_only[(1,)](out, BLOCK=B)
        torch.npu.synchronize()
        ok = torch.equal(out, torch.arange(B, dtype=torch.int32, device="npu"))
        return f"ran, unrelated store {'correct' if ok else 'WRONG'}"

    if CASE == "to_tensor":
        out = torch.zeros(B, dtype=torch.int32, device="npu")
        k_to_tensor[(1,)](out, BLOCK=B)
        torch.npu.synchronize()
        return f"ran, out[:8]={out[:8].tolist()} (UB is uninitialised, value is not the point)"

    if CASE == "copy":
        src = torch.arange(B, dtype=torch.int32, device="npu")
        out = torch.zeros(B, dtype=torch.int32, device="npu")
        k_copy[(1,)](src, out, BLOCK=B)
        torch.npu.synchronize()
        ok = torch.equal(out, src * 2)
        return f"ran, {'CORRECT' if ok else 'WRONG'} out[:8]={out[:8].tolist()}"

    if CASE == "ptr_store":
        out = torch.zeros(B, dtype=torch.int32, device="npu")
        k_ptr_store[(1,)](out, BLOCK=B)
        torch.npu.synchronize()
        ok = torch.equal(out, torch.arange(B, dtype=torch.int32, device="npu") * 3)
        return f"ran, {'CORRECT' if ok else 'WRONG'} out[:8]={out[:8].tolist()}"

    if CASE == "atomic":
        NBIN = 64
        idx = torch.randint(0, NBIN, (B,), dtype=torch.int32, device="npu")
        out = torch.zeros(NBIN, dtype=torch.int32, device="npu")
        k_atomic[(1,)](idx, out, BLOCK=B, NBIN=NBIN)
        torch.npu.synchronize()
        ref = torch.bincount(idx.cpu().long(), minlength=NBIN).to(torch.int32)
        ok = torch.equal(out.cpu(), ref)
        return (f"ran, histogram {'CORRECT' if ok else 'WRONG'} "
                f"| sum={int(out.sum())} expected {B}")

    if CASE.startswith("cap"):
        n = int(CASE[3:])
        out = torch.zeros(n, dtype=torch.int32, device="npu")
        k_cap[(1,)](out, N=n)
        torch.npu.synchronize()
        return f"ran at N={n} ({n * 4 // 1024} KB)"

    return f"!! unknown case {CASE}"


print(f"--- case {CASE} | ASCEND_LAUNCH_BLOCKING="
      f"{os.environ.get('ASCEND_LAUNCH_BLOCKING', 'unset')}")
sys.stdout.flush()
try:
    print(f"RESULT {CASE}: {run()}")
except Exception:
    print(f"RESULT {CASE}: FAILED")
    sys.stdout.flush()
    traceback.print_exc(file=sys.stdout)
sys.stdout.flush()
