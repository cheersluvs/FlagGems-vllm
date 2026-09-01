#!/usr/bin/env python3
"""Which Triton construct does this backend refuse?

On the 910B, top_k_per_row's minimal launch dies with an MLIR assertion inside
the FlagTree-built compiler:

    UseDefLists.h:198 ~IRObjectWithUseList<BlockOperand>():
    Assertion `use_empty() && "Cannot destroy a value that still has uses!"'
    Aborted (core dumped)

That is a compiler defect, not a wrong kernel -- but before blaming the compiler
it has to be shown to compile ANYTHING, and then which construct breaks it.
Probes run from trivial to the exact shape the operator uses, so the first
failure names the construct rather than the operator.

Each probe is a SEPARATE PROCESS. An MLIR assertion aborts, so a probe that
crashes would take every later one with it, and a probe that merely poisons the
context would make every later result meaningless -- the same reason the
preflight refuses to launch two kernels in one run.

    source /usr/local/Ascend/cann/set_env.sh
    PYTHONPATH=src:$PYTHONPATH python tools/triton_smoke.py

Vendor-neutral: nothing here is Ascend-specific, so it serves the next new card
as well. Measurement only.
"""

import os
import subprocess
import sys
import tempfile

HEAD = """
import torch, triton, triton.language as tl
try:
    import torch_npu  # noqa: F401
    DEV = "npu"
except ImportError:
    DEV = "cuda"
N = 1024
"""

TAIL = """
print("OK")
"""

PROBES = [
    ("0. 什么都不编，只分配", """
x = torch.randn(N, device=DEV)
assert float(x.sum()) == float(x.sum())
"""),
    ("1. 逐元素", """
@triton.jit
def k(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i) * 2.0)
x = torch.randn(N, device=DEV); y = torch.empty_like(x)
k[(1,)](x, y, BLOCK=N)
assert torch.allclose(y, x * 2, atol=1e-5)
"""),
    ("2. 归约 tl.max / tl.sum", """
@triton.jit
def k(X, Y, BLOCK: tl.constexpr):
    v = tl.load(X + tl.arange(0, BLOCK))
    tl.store(Y + 0, tl.max(v, axis=0))
    tl.store(Y + 1, tl.sum(v, axis=0))
x = torch.randn(N, device=DEV); y = torch.empty(2, device=DEV)
k[(1,)](x, y, BLOCK=N)
"""),
    ("3. tl.cumsum 2048 宽", """
@triton.jit
def k(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, tl.cumsum(tl.load(X + i), axis=0))
x = torch.randn(2048, device=DEV); y = torch.empty_like(x)
k[(1,)](x, y, BLOCK=2048)
"""),
    ("4. 全局 atomic_add 带 mask", """
@triton.jit
def k(X, H, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    b = (tl.load(X + i) * 8).to(tl.int32) % 64
    tl.atomic_add(H + b, tl.full([BLOCK], 1, tl.int32), mask=i < BLOCK)
x = torch.rand(N, device=DEV); h = torch.zeros(64, dtype=torch.int32, device=DEV)
k[(1,)](x, h, BLOCK=N)
assert int(h.sum()) == N
"""),
    ("5. static_range 循环", """
@triton.jit
def k(X, Y, BLOCK: tl.constexpr, R: tl.constexpr):
    i = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], tl.float32)
    for r in tl.static_range(0, R):
        acc += tl.load(X + i) * r
    tl.store(Y + i, acc)
x = torch.randn(N, device=DEV); y = torch.empty_like(x)
k[(1,)](x, y, BLOCK=N, R=4)
"""),
    ("6. static_range + 数据依赖守卫（算子的形状）", """
@triton.jit
def k(X, Y, BLOCK: tl.constexpr, R: tl.constexpr):
    i = tl.arange(0, BLOCK)
    found = tl.full((), False, dtype=tl.int1)
    acc = tl.zeros([BLOCK], tl.float32)
    for r in tl.static_range(0, R):
        if not found:
            v = tl.load(X + i) + r
            acc += v
            found = tl.max((v > 0.99).to(tl.int32), axis=0) != 0
    tl.store(Y + i, acc)
x = torch.rand(N, device=DEV); y = torch.empty_like(x)
k[(1,)](x, y, BLOCK=N, R=4)
"""),
    ("7. 同上 + 循环内 tl.store 带 mask", """
@triton.jit
def k(X, Y, Z, BLOCK: tl.constexpr, R: tl.constexpr):
    i = tl.arange(0, BLOCK)
    found = tl.full((), False, dtype=tl.int1)
    for r in tl.static_range(0, R):
        if not found:
            v = tl.load(X + i) + r
            m = v > 0.99
            tl.store(Z + i, i.to(tl.int32), mask=m)
            found = tl.max(m.to(tl.int32), axis=0) != 0
    tl.store(Y + i, tl.load(X + i))
x = torch.rand(N, device=DEV); y = torch.empty_like(x)
z = torch.zeros(N, dtype=torch.int32, device=DEV)
k[(1,)](x, y, z, BLOCK=N, R=4)
"""),
    ("8. 同上 + 循环内 debug_barrier", """
@triton.jit
def k(X, Y, BLOCK: tl.constexpr, R: tl.constexpr):
    i = tl.arange(0, BLOCK)
    found = tl.full((), False, dtype=tl.int1)
    for r in tl.static_range(0, R):
        if not found:
            v = tl.load(X + i) + r
            found = tl.max((v > 0.99).to(tl.int32), axis=0) != 0
            tl.debug_barrier()
    tl.store(Y + i, tl.load(X + i))
x = torch.rand(N, device=DEV); y = torch.empty_like(x)
k[(1,)](x, y, BLOCK=N, R=4)
"""),
    ("9. tl.cumsum(reverse=True)  ← 非 TLE 分支实际用的", """
@triton.jit
def k(X, Y, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    tl.store(Y + i, tl.cumsum(tl.load(X + i), axis=0, reverse=True))
x = torch.randn(2048, device=DEV); y = torch.empty_like(x)
k[(1,)](x, y, BLOCK=2048)
"""),
    ("10. 算子的扫描循环原样（cumsum 在守卫内）", """
@triton.jit
def k(H, Y, BLOCK: tl.constexpr, R: tl.constexpr, TOPK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    found = tl.full((), False, dtype=tl.int1)
    last = 0
    for r in tl.static_range(0, R):
        if not found:
            bins = r * BLOCK + lane
            counts = tl.load(H + bins)
            total = tl.sum(counts)
            prefix = total - tl.cumsum(counts, axis=0, reverse=True) + last
            nxt = prefix + counts
            m = (prefix < TOPK) & (nxt >= TOPK)
            tl.store(Y + bins, prefix, mask=m)
            found = tl.max(m.to(tl.int32), axis=0) != 0
            last = total + last
h = torch.randint(0, 4, (2048,), dtype=torch.int32, device=DEV)
y = torch.zeros(2048, dtype=torch.int32, device=DEV)
k[(1,)](h, y, BLOCK=512, R=4, TOPK=64)
"""),
    ("11. 同上但 cumsum 提到守卫外（候选绕法）", """
@triton.jit
def k(H, Y, BLOCK: tl.constexpr, R: tl.constexpr, TOPK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    found = tl.full((), False, dtype=tl.int1)
    last = 0
    for r in tl.static_range(0, R):
        bins = r * BLOCK + lane
        counts = tl.load(H + bins)
        total = tl.sum(counts)
        prefix = total - tl.cumsum(counts, axis=0, reverse=True) + last
        nxt = prefix + counts
        m = (prefix < TOPK) & (nxt >= TOPK) & (not found)
        tl.store(Y + bins, prefix, mask=m)
        found = found | (tl.max(m.to(tl.int32), axis=0) != 0)
        last = total + last
h = torch.randint(0, 4, (2048,), dtype=torch.int32, device=DEV)
y = torch.zeros(2048, dtype=torch.int32, device=DEV)
k[(1,)](h, y, BLOCK=512, R=4, TOPK=64)
"""),
]
