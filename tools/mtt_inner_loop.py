#!/usr/bin/env python3
"""Isolate what the prefill inner loop actually spends per element, on MTT.

Established so far: the gap to vLLM is neither fixed cost (1.16 us, ceiling 3.9%)
nor occupancy nor refinement depth (hard data slows both sides equally). It is
plain per-element throughput -- 1.466 vs 0.856 ns/elem, 1.72x -- and both sides
sit far from the bandwidth roof (476x / 278x), so the inner loop is instruction
bound.

Per element the loop does bin extraction (~6 ops) and then ONE shared-memory
atomic_add into a 2048-bin histogram, from 512 threads. With randn the bins are
Gaussian-concentrated, so hot bins serialise. That is the standing suspect, and
it is measurable without touching the real kernel by running the same loop three
ways:

    A  load + bin extract, no atomic          -> the floor
    B  load + bin extract + atomic (current)  -> B - A is the atomic's cost
    C  same, but NSUB privatised histograms   -> does splitting relieve it?

If B - A is small, atomics are not the problem and the cost is in the extract or
the load, which points somewhere else entirely. If C beats B, privatisation is
the lever and we know roughly what it buys BEFORE writing an override.

    VLLM_PLUGINS=musa PYTHONPATH=src python tools/mtt_inner_loop.py

Measurement only. Writes nothing, proposes nothing.
"""

import sys

import torch
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm.ops.top_k_per_row_prefill import _extract_bin_idx
from flaggems_vllm.utils.triton_version_utils import has_triton_tle

DEV = flaggems_vllm.device
NUM_BINS = 2048
BLOCK = 512

HAS_TLE = False
if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE = True
    except ImportError:
        tle = None


@triton.jit
def _k_extract_only(X, SINK, N, BLOCK: tl.constexpr):
    """A: load + bin extract, no atomic. Accumulates so nothing is dead-coded."""
    pid = tl.program_id(0)
    X += pid * N
    acc = tl.zeros([BLOCK], dtype=tl.uint32)
    for off in tl.range(0, N, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < N
        x = tl.load(X + idx, mask=m, other=0.0)
        bin_idx, _ = _extract_bin_idx(x, m, 0, STEP=0)
        acc += bin_idx
    tl.store(SINK + pid, tl.sum(acc))


@triton.jit
def _k_atomic(X, SINK, N, BLOCK: tl.constexpr, NUM_BINS: tl.constexpr):
    """B: exactly what the real kernel does today."""
    pid = tl.program_id(0)
    X += pid * N
    hist = tle.gpu.alloc(
        [NUM_BINS], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    hp = tle.gpu.local_ptr(hist, (0,))
    for z in tl.range(0, NUM_BINS, BLOCK):
        tl.store(hp + z + tl.arange(0, BLOCK), 0)
    tl.debug_barrier()

    ones = tl.full([BLOCK], 1, tl.int32)
    for off in tl.range(0, N, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = idx < N
        x = tl.load(X + idx, mask=m, other=0.0)
        bin_idx, match = _extract_bin_idx(x, m, 0, STEP=0)
        tl.atomic_add(hp + bin_idx, ones, mask=match, sem="relaxed", scope="cta")
    tl.debug_barrier()

    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for z in tl.range(0, NUM_BINS, BLOCK):
        acc += tl.load(hp + z + tl.arange(0, BLOCK))
    tl.store(SINK + pid, tl.sum(acc))


@triton.jit
def _k_atomic_private(
    X, SINK, N, BLOCK: tl.constexpr, NUM_BINS: tl.constexpr, NSUB: tl.constexpr
):
    """C: NSUB private histograms, lane-strided, reduced at the end.

    Splits contention NSUB ways at NSUB x the shared memory.
    """
    pid = tl.program_id(0)
    X += pid * N
    hist = tle.gpu.alloc(
        [NSUB * NUM_BINS], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    hp = tle.gpu.local_ptr(hist, (0,))
    for z in tl.range(0, NSUB * NUM_BINS, BLOCK):
        tl.store(hp + z + tl.arange(0, BLOCK), 0)
    tl.debug_barrier()

    lane = tl.arange(0, BLOCK)
    sub = (lane % NSUB) * NUM_BINS
    ones = tl.full([BLOCK], 1, tl.int32)
    for off in tl.range(0, N, BLOCK):
        idx = off + lane
        m = idx < N
        x = tl.load(X + idx, mask=m, other=0.0)
        bin_idx, match = _extract_bin_idx(x, m, 0, STEP=0)
        tl.atomic_add(hp + sub + bin_idx, ones, mask=match, sem="relaxed", scope="cta")
    tl.debug_barrier()

    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for z in tl.range(0, NSUB * NUM_BINS, BLOCK):
        acc += tl.load(hp + z + tl.arange(0, BLOCK))
    tl.store(SINK + pid, tl.sum(acc))


def _bench(fn):
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")


def run(num_rows, N, label, logits):
    sink = torch.empty((num_rows,), dtype=torch.int32, device=DEV)
    nw = BLOCK // 32

    a = _bench(
        lambda: _k_extract_only[(num_rows,)](
            logits, sink, N, BLOCK=BLOCK, num_warps=nw
        )
    )
    b = _bench(
        lambda: _k_atomic[(num_rows,)](
            logits, sink, N, BLOCK=BLOCK, NUM_BINS=NUM_BINS, num_warps=nw
        )
    )
    print(f"\n  --- {label} ---")
    ns = 1e6 / (num_rows * N)
    print(f"    A extract only     {a*1000:8.2f} us   {a*ns*1e3:.4f} ns/elem")
    print(f"    B + atomic (今天)  {b*1000:8.2f} us   {b*ns*1e3:.4f} ns/elem")
    print(f"    => atomic 占 {(b-a)/b*100:5.1f}%  ({(b-a)*1000:.2f} us)")

    best, bestn = b, 1
    for nsub in (2, 4, 8):
        smem = nsub * NUM_BINS * 4
        if smem > 96 * 1024:
            print(f"    C NSUB={nsub:<2} 跳过 (需 {smem//1024} KB smem)")
            continue
        try:
            c = _bench(
                lambda ns=nsub: _k_atomic_private[(num_rows,)](
                    logits, sink, N, BLOCK=BLOCK, NUM_BINS=NUM_BINS, NSUB=ns,
                    num_warps=nw,
                )
            )
        except Exception as e:  # noqa: BLE001
            print(f"    C NSUB={nsub:<2} 失败: {type(e).__name__}: {str(e)[:60]}")
            continue
        tag = "  <-- 更好" if c < best else ""
        if c < best:
            best, bestn = c, nsub
        print(
            f"    C NSUB={nsub:<2} ({smem//1024:>2} KB)   {c*1000:8.2f} us"
            f"   vs B {b/c:5.2f}x{tag}"
        )
    if bestn > 1:
        print(f"    => 私有化最好 NSUB={bestn}, 比 B 快 {b/best:.2f}x")
    else:
        print("    => 私有化没有帮助")


def main():
    print("=" * 78)
    print("  MTT prefill 内层循环拆解 -- 仅测量")
    print("=" * 78)
    if not HAS_TLE:
        print("  !! 无 TLE，无法分配 smem 直方图，退出")
        return 1
    print(f"  device {DEV}   BLOCK={BLOCK}   NUM_BINS={NUM_BINS}")
    print("\n  参考: 整算子 1.466 ns/elem, vLLM 0.856 ns/elem (差 1.72x)")

    num_rows, N = 60, 131072
    torch.manual_seed(42)
    run(num_rows, N, "randn (真实分布)", torch.randn((num_rows, N), device=DEV))

    # Uniform bins: if contention is the cost, a flat bin distribution should be
    # markedly cheaper than the Gaussian one at identical instruction count.
    u = torch.randint(0, 2**31 - 1, (num_rows, N), dtype=torch.int32, device=DEV)
    run(num_rows, N, "均匀 bin (对照: 无热点)", u.view(torch.float32))
    print("\n  两者 B 差距大 => 是热点争用; 差距小 => 是原子指令本身的固定成本")
    return 0


if __name__ == "__main__":
    sys.exit(main())
