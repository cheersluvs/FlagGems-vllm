"""Standalone reproducers for the MetaX TLE (mctle) defects, one per fix.

Only torch + triton + triton.experimental.tle. Each case runs in its own
process: case D aborts the interpreter on every build known so far.

    python tools/flagtree_metax_repro.py          # all cases
    python tools/flagtree_metax_repro.py A        # one case

    case  needs                                     symptom when missing
    A     -D__MCTLE__ for TableGen                  tt.atomic_rmw rejected
          (cmake/FlagTreeOptions.cmake appends it   by the verifier
           to LLVM_TABLEGEN_FLAGS)
    B     third_party/metax/lib/Analysis/Alias.cpp  buffers silently share
          local_pointers aliasing                   bytes -> wrong values
    C     the same Alias.cpp fix                    tl.histogram's scratch
                                                    lands on a TLE buffer
                                                    (FlagTree issue #1152,
                                                     Problem 4)
    D     nothing here fixes it -- prebuilt plugin  assertion, process abort
          (FlagTree issue #1152, Problem 1)

Both fixes are on https://github.com/cheersluvs/FlagTree branch
metax-mctle-tle-fixes, on top of 0.6.1+metax3.6 + #971.
"""

import os
import subprocess
import sys

CASES = {
    "A": "smem atomic_add verifies and counts correctly",
    "B": "two smem buffers written through local_ptr keep their own bytes",
    "C": "smem round-trip with tl.histogram in between (issue #1152 L7)",
    "D": "scalar local_ptr + offsets, 1 element/thread (plugin assert)",
}

if len(sys.argv) == 1:
    print(f"python {sys.executable}\n")
    for name, what in CASES.items():
        r = subprocess.run(
            [sys.executable, os.path.abspath(__file__), name],
            capture_output=True, text=True, timeout=900,
        )
        out = (r.stdout + r.stderr).splitlines()
        line = next((x for x in out if x.startswith("RESULT")), None)
        if line is None:
            hint = next(
                (x for x in out if "Assertion" in x or "error:" in x or "Error" in x),
                out[-1] if out else "no output",
            )
            line = f"RESULT {name}: ABORTED (exit {r.returncode}) -- {hint.strip()[:150]}"
        print(f"{line}\n    {what}")
    sys.exit(0)

CASE = sys.argv[1]

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402
import triton.experimental.tle.language as tle  # noqa: E402

N = 512
WARPS = 8
TILE = 4096


def alloc(n):
    return tle.gpu.alloc(
        [n], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )


@triton.jit
def k_atomic(idx_ptr, out_ptr, N: tl.constexpr):
    """Case A: masked atomic_add on a shared-memory pointer, the form a
    radix/top-k histogram uses. Without __MCTLE__ in TableGen, metax's
    TritonOps.td keeps the #else constraint (getPointerTypeSameShape, address
    space 1) and this fails verification with
    'tt.atomic_rmw op failed to verify that ptr type matches value type'."""
    buf = tle.gpu.alloc([N], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
                        nv_mma_shared_layout=False)
    view = tle.gpu.local_ptr(buf)
    lane = tl.arange(0, N)
    tl.store(view, tl.zeros([N], tl.int32))
    tl.debug_barrier()
    scalar = tle.gpu.local_ptr(buf, (0,))
    b = tl.load(idx_ptr + lane)
    tl.atomic_add(scalar + b, lane * 0 + 1, mask=b >= 0, sem="relaxed", scope="cta")
    tl.debug_barrier()
    tl.store(out_ptr + lane, tl.load(view))


@triton.jit
def k_two_buffers(out_ptr, N: tl.constexpr):
    """Case B: buffer A is written only through the pointer local_ptr gives
    back, and B is allocated after A's last DIRECT use. Without the alias fix
    A looks dead at that point, B is given A's offset, and the cross-warp
    reduction scratch is laid over both."""
    lane = tl.arange(0, N)
    a = tle.gpu.alloc([N], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
                      nv_mma_shared_layout=False)
    pa = tle.gpu.local_ptr(a, (0,)) + (tl.program_id(0) >> 31)
    tl.store(pa + lane, lane)
    b = tle.gpu.alloc([N], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
                      nv_mma_shared_layout=False)
    pb = tle.gpu.local_ptr(b, (0,)) + (tl.program_id(0) >> 31)
    tl.store(pb + lane, lane + N)
    total = tl.sum(lane, axis=0)          # cross-warp reduction: needs scratch
    run = tl.cumsum(lane, axis=0)         # and so does this
    tl.debug_barrier()
    bad = tl.sum((tl.load(pa + lane) != lane).to(tl.int32), axis=0)
    bad += tl.sum((tl.load(pb + lane) != lane + N).to(tl.int32), axis=0)
    bad += (total != N * (N - 1) // 2).to(tl.int32)
    bad += tl.sum((run != tl.cumsum(lane, axis=0)).to(tl.int32), axis=0)
    tl.store(out_ptr + lane, bad)


@triton.jit
def k_histogram(in_ptr, out_ptr, TILE: tl.constexpr, BINS: tl.constexpr):
    """Case C: FlagTree issue #1152 Listing 7, shortened -- store a tile to
    smem, run tl.histogram, read the tile back. tl.histogram's scratch and the
    TLE buffer are allocated on top of each other."""
    lane = tl.arange(0, TILE)
    buf = tle.gpu.alloc([TILE], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
                        nv_mma_shared_layout=False)
    p = tle.gpu.local_ptr(buf, (lane,))
    v = tl.load(in_ptr + lane)
    tl.store(p, v)
    tl.debug_barrier()
    h = tl.histogram(v % BINS, BINS)
    tl.debug_barrier()
    back = tl.load(tle.gpu.local_ptr(buf, (lane,)))
    tl.store(out_ptr + lane, back + tl.sum(h, axis=0) * 0)


@triton.jit
def k_scalar_offsets(out_ptr, N: tl.constexpr):
    """Case D: FlagTree issue #1152 Problem 1. The plugin's __MCTLE__ block
    widens the vector width of an UNMASKED shared load from pointer alignment
    alone, unclamped by elements per thread, then asserts
    `wordNElems * nWords * numVecs == numElems`. Measured boundary on
    0.6.1+metax3.6 + #971: >= 4 elements per thread passes, fewer aborts.
    Not fixed by either FlagTree patch -- it is inside the prebuilt plugin."""
    buf = tle.gpu.alloc([N], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
                        nv_mma_shared_layout=False)
    p = tle.gpu.local_ptr(buf, (0,))
    lane = tl.arange(0, N)
    tl.store(tle.gpu.local_ptr(buf), lane)
    tl.debug_barrier()
    tl.store(out_ptr + lane, tl.load(p + lane))


dev = "cuda"
torch.manual_seed(0)

if CASE == "A":
    idx = torch.randint(0, N, (N,), dtype=torch.int32, device=dev)
    out = torch.zeros(N, dtype=torch.int32, device=dev)
    try:
        k_atomic[(1,)](idx, out, N=N, num_warps=WARPS)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        inner = str(e).strip().splitlines()[-1][:140]
        print(f"RESULT A: FAILED to compile -- {type(e).__name__}: {inner}")
        sys.exit(1)
    want = torch.bincount(idx.cpu().long(), minlength=N).to(torch.int32)
    ok = torch.equal(out.cpu(), want)
    print(f"RESULT A: {'PASS' if ok else 'WRONG'} (sum={int(out.sum())}, want {N})")
elif CASE == "B":
    out = torch.full((N,), -1, dtype=torch.int32, device=dev)
    ck = k_two_buffers[(1,)](out, N=N, num_warps=WARPS)
    torch.cuda.synchronize()
    bad = int(out[0].item())
    shared = getattr(getattr(ck, "metadata", None), "shared", -1)
    need = 2 * N * 4
    ok = bad == 0 and shared >= need
    print(f"RESULT B: {'PASS' if ok else 'WRONG'} (mismatching values={bad}, "
          f"shared={shared} B for {need} B of buffers + reduction scratch)")
elif CASE == "C":
    x = torch.randint(0, 1 << 16, (TILE,), dtype=torch.int32, device=dev)
    out = torch.full((TILE,), -1, dtype=torch.int32, device=dev)
    k_histogram[(1,)](x, out, TILE=TILE, BINS=256, num_warps=WARPS)
    torch.cuda.synchronize()
    bad = int((out != x).sum())
    print(f"RESULT C: {'PASS' if bad == 0 else 'WRONG'} ({bad} of {TILE} values "
          f"corrupted by the histogram's scratch)")
elif CASE == "D":
    out = torch.zeros(N, dtype=torch.int32, device=dev)
    k_scalar_offsets[(1,)](out, N=N, num_warps=WARPS)
    torch.cuda.synchronize()
    ok = torch.equal(out, torch.arange(N, dtype=torch.int32, device=dev))
    print(f"RESULT D: {'PASS' if ok else 'WRONG'} "
          f"(N={N}, {N // (WARPS * 64)} elements/thread)")
else:
    print(f"unknown case {CASE}")
    sys.exit(2)
