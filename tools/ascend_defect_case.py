"""One backend defect per process: is it still there on this toolchain?

Every case below is a workaround that currently lives in
runtime/backend/_ascend/fused/top_k_per_row_*.py.  A case that PASSES means the
defect is gone and that workaround can be deleted; a case that FAILS means the
workaround is still earning its place.  This is the only question worth asking
of a new FlagTree build first -- "does it run" says nothing about how much of
the override is now dead weight.

Run one per process (tools/ascend_dsa_run.py), because a device fault poisons
every later launch in the same process.
"""

# torch_npu must be imported explicitly, before triton.  On FlagTree 0.6.1 the
# import graph is circular under torch's automatic backend loading: triton pulls
# in torch, torch auto-loads torch_npu, torch_npu re-enters triton, and triton
# dies with "cannot import name 'backends' from partially initialized module".
# TORCH_DEVICE_BACKEND_AUTOLOAD=0 (set by the verify script) disables the
# autoload; this import is what replaces it.
import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    pass
try:
    import torch_musa  # noqa: F401
except Exception:
    pass

import triton
import triton.language as tl

import sys
import traceback


def _device():
    for name in ("npu", "musa", "cuda"):
        mod = getattr(torch, name, None)
        if mod is not None and mod.is_available():
            return name, mod
    raise RuntimeError("no accelerator found")


DEV, DEVMOD = _device()
CASE = sys.argv[1]
B = 128


# 1 -------------------------------------------------------------- tl.reduce_or
@triton.jit
def k_reduce_or(x_ptr, out_ptr, BLOCK: tl.constexpr):
    m = tl.load(x_ptr + tl.arange(0, BLOCK)) != 0
    tl.store(out_ptr, tl.reduce_or(m, axis=0).to(tl.int32))


# 2 ----------------------------------------------------------------- tl.assume
@triton.jit
def k_assume(n, out_ptr, BLOCK: tl.constexpr):
    tl.assume(n > 0)
    tl.store(out_ptr + tl.arange(0, BLOCK), tl.arange(0, BLOCK))


# 3 ------------------------------------------ uint16 >> must be LOGICAL, not arithmetic
@triton.jit
def k_uint16_shift(x_ptr, out_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    bits = tl.load(x_ptr + lane).to(tl.uint16, bitcast=True)
    tl.store(out_ptr + lane, (bits >> 5).to(tl.int32))


# 4 ------------------------------------- atomic_add must return unique per-lane olds
@triton.jit
def k_atomic_unique(cnt_ptr, out_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    tl.store(out_ptr + lane, tl.atomic_add(cnt_ptr + tl.zeros([BLOCK], tl.int32), 1))


# 5 --------------------------------- masked store with DUPLICATE lane addresses
@triton.jit
def k_dup_store(out_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    tl.store(out_ptr + (lane // 8), lane, mask=lane % 8 == 0)


# 6 ------------------------------------ load mask with a runtime ROW offset
@triton.jit
def k_row_mask(x_ptr, row, n, out_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    m = lane < n
    tl.store(out_ptr + lane, tl.load(x_ptr + row * BLOCK + lane, mask=m, other=-1))


# 7 ------------------------------------------ tl.where inside a loop (UB allocator)
@triton.jit
def k_where_in_loop(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.int32)
    for i in tl.range(0, N):
        v = tl.load(x_ptr + i * BLOCK + lane)
        acc += tl.where(v > 0, v, 0)
    tl.store(out_ptr + lane, acc)


# 8 ------------------------------------------------ reshape a 2-D tile to 1-D
@triton.jit
def k_reshape_2d(x_ptr, out_ptr, BLOCK: tl.constexpr, VEC: tl.constexpr):
    off = tl.arange(0, BLOCK)[:, None] * VEC + tl.arange(0, VEC)[None, :]
    tl.store(out_ptr + tl.arange(0, BLOCK * VEC),
             tl.reshape(tl.load(x_ptr + off), [BLOCK * VEC]))


def run():
    if CASE == "reduce_or":
        x = torch.zeros(B, dtype=torch.int32, device=DEV); x[7] = 1
        out = torch.zeros(1, dtype=torch.int32, device=DEV)
        k_reduce_or[(1,)](x, out, BLOCK=B); DEVMOD.synchronize()
        return f"reduce_or lowers, got {int(out[0])} (expected 1)"

    if CASE == "assume":
        out = torch.zeros(B, dtype=torch.int32, device=DEV)
        k_assume[(1,)](8, out, BLOCK=B); DEVMOD.synchronize()
        ok = torch.equal(out, torch.arange(B, dtype=torch.int32, device=DEV))
        return f"tl.assume lowers, store {'correct' if ok else 'WRONG'}"

    if CASE == "uint16_shift":
        # negative int16 -> high bit set; a logical >> 5 must clear the top bits
        x = torch.full((B,), -2, dtype=torch.int16, device=DEV)
        out = torch.zeros(B, dtype=torch.int32, device=DEV)
        k_uint16_shift[(1,)](x, out, BLOCK=B); DEVMOD.synchronize()
        got, exp = int(out[0]), (0xFFFE >> 5)
        return (f"{'LOGICAL (fixed)' if got == exp else 'ARITHMETIC (defect)'}"
                f" got {got}, logical shift would give {exp}")

    if CASE == "atomic_unique":
        cnt = torch.zeros(1, dtype=torch.int32, device=DEV)
        out = torch.zeros(B, dtype=torch.int32, device=DEV)
        k_atomic_unique[(1,)](cnt, out, BLOCK=B); DEVMOD.synchronize()
        u = len(set(out.cpu().tolist()))
        return (f"{'unique (fixed)' if u == B else 'NOT unique (defect)'}"
                f": {u}/{B} distinct, counter={int(cnt[0])}")

    if CASE == "dup_store":
        out = torch.full((B // 8,), -1, dtype=torch.int32, device=DEV)
        k_dup_store[(1,)](out, BLOCK=B); DEVMOD.synchronize()
        dropped = int((out == -1).sum())
        return (f"{'all slots written (fixed)' if dropped == 0 else 'DROPPED (defect)'}"
                f": {dropped}/{B // 8} untouched")

    if CASE == "row_mask":
        rows, n = 4, 40
        x = torch.arange(rows * B, dtype=torch.int32, device=DEV)
        out = torch.zeros(B, dtype=torch.int32, device=DEV)
        k_row_mask[(1,)](x, 2, n, out, BLOCK=B); DEVMOD.synchronize()
        exp = torch.full((B,), -1, dtype=torch.int32)
        exp[:n] = torch.arange(2 * B, 2 * B + n, dtype=torch.int32)
        ok = torch.equal(out.cpu(), exp)
        return (f"{'correct (fixed)' if ok else 'WRONG DATA (defect)'}"
                f": out[0]={int(out[0])} expected {2 * B}")

    if CASE == "where_in_loop":
        N = 8
        x = torch.arange(N * B, dtype=torch.int32, device=DEV) - (N * B // 2)
        out = torch.zeros(B, dtype=torch.int32, device=DEV)
        k_where_in_loop[(1,)](x, out, N, BLOCK=B); DEVMOD.synchronize()
        exp = x.reshape(N, B).clamp(min=0).sum(0)
        ok = torch.equal(out, exp)
        return f"compiles and {'CORRECT' if ok else 'WRONG'}"

    if CASE == "reshape_2d":
        VEC = 4
        x = torch.arange(B * VEC, dtype=torch.int32, device=DEV)
        out = torch.zeros(B * VEC, dtype=torch.int32, device=DEV)
        k_reshape_2d[(1,)](x, out, BLOCK=B, VEC=VEC); DEVMOD.synchronize()
        ok = torch.equal(out, x)
        return f"compiles and {'CORRECT' if ok else 'WRONG'}"

    return f"!! unknown case {CASE}"


print(f"--- defect case {CASE} | {DEV} | triton {triton.__version__}")
sys.stdout.flush()
try:
    print(f"RESULT {CASE}: {run()}")
except Exception:
    print(f"RESULT {CASE}: FAILED (defect still present, or a new one)")
    sys.stdout.flush()
    traceback.print_exc(file=sys.stdout)
sys.stdout.flush()
