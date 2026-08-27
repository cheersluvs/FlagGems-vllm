#!/usr/bin/env python3
"""Count how many refinement passes prefill actually makes, per row.

This exists because an earlier measurement of mine was wrong in a way that
inverted a conclusion. I timed the inner loop with SCALAR loads and got 252 GB/s,
concluded the load dominated a single pass, and therefore that the operator was
essentially one pass and refinement depth was not a lever.

The kernel does not load like that. Its scan builds `base = t*BLOCK*VEC +
lane*VEC` with VEC=4 and loads a [BLOCK, 4] tile. Measured with that pattern the
load runs at 645-661 GB/s at 60-64 rows -- 2.5x what I reported -- so one full
histogram pass costs about 69 us at (60, 131072), not 177. Against an operator
time of 209 us that is roughly THREE passes, and vLLM's 129 us is roughly two.

If that holds, pass count is where the 1.63x lives, and the rewrite worth trying
is making step 0 selective enough to finish sooner -- not touching the loop body,
which is already vectorised, with near-free bit extraction and an atomic worth
16%.

But "roughly three" is inferred from ratios. This counts it directly: the same
four-step loop, with a counter incremented on each step that actually executes.

    VLLM_PLUGINS=musa PYTHONPATH=src python tools/mtt_pass_count.py

Measurement only.
"""

import sys

import torch
import triton
import triton.language as tl

import flaggems_vllm
from flaggems_vllm.ops.top_k_per_row_prefill import (
    NUM_THREADS_PER_BLOCK,
    _distribute_to_bins,
    _extract_bin_idx,
    _process_bins,
    _wide_block_max_rows,
    _num_warps,
    _process_histogram_step,
)

DEV = flaggems_vllm.device

try:
    import triton.experimental.tle.language as tle
except ImportError:
    tle = None


@triton.jit
def _k_scan_only(
    X, SINK, N, BLOCK: tl.constexpr, NUM_BINS: tl.constexpr, VEC: tl.constexpr
):
    """Histogram clear + the REAL vec4 scan + the atomic. No _process_bins.

    Mirrors the operator's aligned path exactly -- `base = t*BLOCK*VEC +
    lane*VEC`, a [BLOCK, VEC] tile, `_distribute_to_bins(..., STEP=0)` -- so the
    difference against the instrumented full histogram step isolates
    _process_bins, the 2048-bin threshold scan and candidate compaction, which is
    the last component never measured on its own.
    """
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

    lane = tl.arange(0, BLOCK)
    vec = tl.arange(0, VEC)
    ones_vec_2d = tl.full([BLOCK, VEC], 1, tl.int32)
    n_vec_full = N // (BLOCK * VEC)
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK * VEC + lane * VEC
        offs = base[:, None] + vec[None, :]
        x_vec = tl.load(X + offs)
        _distribute_to_bins(x_vec, True, ones_vec_2d, 0, hp, STEP=0)
    tl.debug_barrier()

    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for z in tl.range(0, NUM_BINS, BLOCK):
        acc += tl.load(hp + z + tl.arange(0, BLOCK))
    tl.store(SINK + pid, tl.sum(acc))


@triton.jit
def _k_passB_floor(X, SINK, N, BLOCK: tl.constexpr, VEC: tl.constexpr):
    """Pass B's floor: the same loads and bin extraction, nothing else.

    Accumulates bin_idx so the loop cannot be dead-coded -- with both
    _process_bins flags off there is no store at all, and the compiler deleted
    the entire loop, which is how the previous version reported 3.4 TB/s against
    a 1.3 TB/s roof.
    """
    pid = tl.program_id(0)
    X += pid * N
    lane = tl.arange(0, BLOCK)
    vec = tl.arange(0, VEC)
    acc = tl.zeros([BLOCK, VEC], dtype=tl.uint32)
    n_vec = N // (BLOCK * VEC)
    for t in tl.range(0, n_vec):
        base = t * BLOCK * VEC + lane * VEC
        offs = base[:, None] + vec[None, :]
        b, _ = _extract_bin_idx(tl.load(X + offs), True, 0, STEP=0)
        acc += b
    tail = n_vec * BLOCK * VEC
    acc1 = tl.zeros([BLOCK], dtype=tl.uint32)
    for t in tl.range(0, tl.cdiv(N - tail, BLOCK)):
        offs = tail + t * BLOCK + lane
        m = offs < N
        b, _ = _extract_bin_idx(tl.load(X + offs, mask=m, other=0.0), m, 0, STEP=0)
        acc1 += tl.where(m, b, 0)
    tl.store(SINK + pid, (tl.sum(tl.sum(acc, axis=1), axis=0) + tl.sum(acc1)))


@triton.jit
def _k_passB(
    X, SINK, N, THRS,
    BLOCK: tl.constexpr, NUM_BINS: tl.constexpr, VEC: tl.constexpr,
    WRITE_DIRECTLY: tl.constexpr, USE_FINAL: tl.constexpr,
):
    """Pass B via the REAL _process_bins, with a REAL per-row threshold.

    THRS holds the threshold bin each row would actually reach, computed on the
    host from the same bin mapping. The previous version passed a made-up 1024,
    which made take_lt fire for about half the elements instead of the ~0.8% the
    operator sees, and the atomic cost it reported was that artefact.

    Covers the tail as well: n_vec_full alone leaves 0% of a 1025-wide row and
    50% of a 4095-wide one, which is why those rows previously came out flat.
    """
    pid = tl.program_id(0)
    X += pid * N
    thr = tl.load(THRS + pid)
    hist = tle.gpu.alloc(
        [NUM_BINS], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    fin = tle.gpu.alloc(
        [2048], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    oi = tle.gpu.alloc(
        [2048], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    c1 = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    c2 = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    hp = tle.gpu.local_ptr(hist, (0,))
    fp = tle.gpu.local_ptr(fin, (0,))
    op = tle.gpu.local_ptr(oi, (0,))
    p1 = tle.gpu.local_ptr(c1, (0,))
    p2 = tle.gpu.local_ptr(c2, (0,))
    tl.store(p1, 0)
    tl.store(p2, 0)
    tl.debug_barrier()

    lane = tl.arange(0, BLOCK)
    vec = tl.arange(0, VEC)
    ones2 = tl.full([BLOCK, VEC], 1, tl.int32)
    ones1 = tl.full([BLOCK], 1, tl.int32)
    z2 = tl.zeros([BLOCK, VEC], dtype=tl.int32)
    z1 = tl.zeros([BLOCK], dtype=tl.int32)

    n_vec = N // (BLOCK * VEC)
    for t in tl.range(0, n_vec):
        base = t * BLOCK * VEC + lane * VEC
        offs = base[:, None] + vec[None, :]
        _process_bins(
            tl.load(X + offs), True, ones2, offs, p1 + z2, p2 + z2, 0, thr,
            WRITE_DIRECTLY, USE_FINAL, 0, None, hp, fp, op, None,
            STEP=0, TOPK=1024, MULTIPLE_BLOCKS_PER_ROW=False, MERGE_BLOCKS=False,
        )
    tail = n_vec * BLOCK * VEC
    for t in tl.range(0, tl.cdiv(N - tail, BLOCK)):
        offs = tail + t * BLOCK + lane
        m = offs < N
        _process_bins(
            tl.load(X + offs, mask=m, other=0.0), m, ones1, offs,
            p1 + z1, p2 + z1, 0, thr,
            WRITE_DIRECTLY, USE_FINAL, 0, None, hp, fp, op, None,
            STEP=0, TOPK=1024, MULTIPLE_BLOCKS_PER_ROW=False, MERGE_BLOCKS=False,
        )
    tl.debug_barrier()
    tl.store(SINK + pid, tl.load(p1) + tl.load(p2))


@triton.jit
def _k_passB_agg(
    X, SINK, N, THRS,
    BLOCK: tl.constexpr, VEC: tl.constexpr, USE_FINAL: tl.constexpr,
):
    """Pass B with tile-aggregated atomics instead of per-lane masked ones.

    The operator's two counters are broadcast scalars -- found_ptrs_vec_2d is
    `s_found_topk_values_ptr + zeros_vec_2d` -- so all BLOCK*VEC lanes contend on
    one address every tile, even though the mask only lets ~0.8% of them count.
    Measured cost of the two paths: 73 us, 55% of Pass B and 34% of the operator.

    Here each tile instead sums its hits, takes ONE atomic for the whole tile,
    and derives per-lane slots from an exclusive prefix sum. 2048 atomics become
    one, at the cost of a cumsum over the flattened tile.
    """
    pid = tl.program_id(0)
    X += pid * N
    thr = tl.load(THRS + pid)
    out = tle.gpu.alloc(
        [2048], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    fin = tle.gpu.alloc(
        [2048], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    c1 = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    c2 = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    op_ = tle.gpu.local_ptr(out, (0,))
    fp = tle.gpu.local_ptr(fin, (0,))
    p1 = tle.gpu.local_ptr(c1, (0,))
    p2 = tle.gpu.local_ptr(c2, (0,))
    tl.store(p1, 0)
    tl.store(p2, 0)
    tl.debug_barrier()

    lane = tl.arange(0, BLOCK)
    vec = tl.arange(0, VEC)
    W: tl.constexpr = BLOCK * VEC
    n_vec = N // W
    for t in tl.range(0, n_vec):
        base = t * W + lane * VEC
        offs = base[:, None] + vec[None, :]
        x = tl.load(X + offs)
        b, ok = _extract_bin_idx(x, True, 0, STEP=0)

        take = ok & (b < thr)
        ti = tl.reshape(take.to(tl.int32), [W])
        cnt = tl.sum(ti)
        start = tl.atomic_add(p1, cnt, sem="relaxed", scope="cta")
        pos = tl.reshape(tl.cumsum(ti, axis=0) - ti, [BLOCK, VEC]) + start
        tl.store(op_ + (pos % 2048), tl.reshape(tl.reshape(offs, [W]), [BLOCK, VEC]),
                 mask=take)

        if USE_FINAL:
            takef = ok & (b == thr)
            tf = tl.reshape(takef.to(tl.int32), [W])
            cntf = tl.sum(tf)
            startf = tl.atomic_add(p2, cntf, sem="relaxed", scope="cta")
            posf = tl.reshape(tl.cumsum(tf, axis=0) - tf, [BLOCK, VEC]) + startf
            tl.store(fp + (posf % 2048), x, mask=takef)
    tl.debug_barrier()
    tl.store(SINK + pid, tl.load(p1) + tl.load(p2))


def _real_thresholds(logits, top_k):
    """The threshold bin each row actually reaches, via the kernel's own mapping.

    STEP 0 maps f32 -> f16 -> monotonic u16 -> >>5, and take_lt selects
    bin_idx < threshold, so a lower bin is a larger value: the threshold is the
    smallest b whose prefix count reaches top_k.
    """
    h = logits.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    mapped = torch.where(h & 0x8000 != 0, h, (~h) & 0x7FFF)
    b = (mapped >> 5).clamp_(0, 2047)
    out = torch.empty(logits.shape[0], dtype=torch.int32, device=logits.device)
    for i in range(logits.shape[0]):
        c = torch.bincount(b[i].flatten(), minlength=2048).cumsum(0)
        idx = torch.nonzero(c >= top_k)
        out[i] = int(idx[0]) if idx.numel() else 2047
    return out


@triton.jit
def _count_steps(
    logits_ptr,
    out_indices_ptr,
    row_starts,
    row_ends,
    STEPS,
    stride0,
    stride1,
    vocab_size,
    TOPK: tl.constexpr,
    TOPKP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """The real four-step loop, instrumented. Mirrors tle_top_k_per_row_prefill's
    prologue exactly so the step decisions are the ones the operator makes."""
    NUM_FILNAL_ITEMS: tl.constexpr = 2048
    NUM_BINS: tl.constexpr = 2048
    VEC: tl.constexpr = 4

    row_id = tl.program_id(0)
    row_start = tl.load(row_starts + row_id)
    row_end = tl.load(row_ends + row_id)
    logits_ptr += row_id * stride0
    x_off_mod = (row_id * stride0 + row_start) % VEC
    skip_elems = 0 if x_off_mod == 0 else VEC - x_off_mod
    out_indices_ptr += row_id * TOPK

    s_histogram = tle.gpu.alloc(
        [NUM_BINS], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_logits = tle.gpu.alloc(
        [NUM_FILNAL_ITEMS], dtype=tl.float32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_out_indices = tle.gpu.alloc(
        [TOPKP], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_final_cnt = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_thr = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_bin = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_found = tle.gpu.alloc(
        [1], dtype=tl.int32, layout=None, scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    hp = tle.gpu.local_ptr(s_histogram, (0,))
    flp = tle.gpu.local_ptr(s_final_logits, (0,))
    oip = tle.gpu.local_ptr(s_out_indices, (0,))
    fcp = tle.gpu.local_ptr(s_final_cnt, (0,))
    tbp = tle.gpu.local_ptr(s_thr, (0,))
    fbp = tle.gpu.local_ptr(s_bin, (0,))
    fvp = tle.gpu.local_ptr(s_found, (0,))

    tl.store(fcp, 0)
    tl.store(fvp, 0)
    tl.debug_barrier()

    assume_aligned = (
        (row_start == 0) & (row_end == vocab_size) & (stride1 == 1)
        & ((vocab_size % BLOCK_SIZE) == 0)
    )
    pattern = tl.zeros((), dtype=tl.uint32)
    go = tl.full((), True, dtype=tl.int1)
    thr_bin = tl.full((), -1, dtype=tl.int32)
    n = tl.zeros((), dtype=tl.int32)
    for step_idx in tl.static_range(0, 4):
        if go:
            n += 1
            (go, pattern, thr_bin) = _process_histogram_step(
                logits_ptr, row_start, row_end, stride1, vocab_size, skip_elems,
                None, pattern, thr_bin, assume_aligned,
                hp, flp, fcp, tbp, fbp, fvp, oip, None,
                STEP=step_idx, TOPK=TOPK, BLOCK_SIZE=BLOCK_SIZE, HAS_TLE=True,
                MULTIPLE_BLOCKS_PER_ROW=False, MERGE_BLOCKS=False,
            )
    tl.store(STEPS + row_id, n)


def main():
    print("=" * 78)
    print("  每行实际执行的精化遍数")
    print("=" * 78)
    if tle is None:
        print("  !! 无 TLE, 退出")
        return 1

    # (num_rows, vocab, top_k, note)
    CASES = [
        (60, 131072, 512, "sweep 1 的形状"),
        (64, 129280, 1024, "最差形状"),
        (4100, 1025, 512, "已经赢的形状"),
        (16383, 4095, 512, "赢得最多的形状"),
    ]
    print(f"  {'shape':>16}{'平均遍数':>10}{'最少':>7}{'最多':>7}   备注")
    for rows, vocab, top_k, note in CASES:
        torch.manual_seed(42)
        logits = torch.randn((rows, vocab), dtype=torch.float32, device=DEV)
        starts = torch.zeros((rows,), dtype=torch.int32, device=DEV)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=DEV)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=DEV)
        steps = torch.zeros((rows,), dtype=torch.int32, device=DEV)
        try:
            _count_steps[(rows,)](
                logits, idx, starts, ends, steps,
                logits.stride(0), logits.stride(1), vocab,
                TOPK=top_k, TOPKP=triton.next_power_of_2(top_k),
                BLOCK_SIZE=NUM_THREADS_PER_BLOCK,
                num_warps=_num_warps(NUM_THREADS_PER_BLOCK),
            )
            flaggems_vllm.runtime.torch_device_fn.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"  {f'({rows},{vocab})':>16}   失败 {type(e).__name__}: {str(e)[:36]}")
            continue
        f = steps.float()
        print(f"  {f'({rows},{vocab})':>16}{f.mean().item():>10.2f}"
              f"{int(steps.min()):>7}{int(steps.max()):>7}   {note}", flush=True)

    # --- 直接测量：直方图阶段 vs 整算子 ---
    print("\n" + "=" * 78)
    print("  直方图阶段 vs 整算子 -- 两者都实测, final select 由差值得出")
    print("=" * 78)
    print(f"  {'shape':>16}{'直方图 us':>11}{'整算子 us':>11}{'之后 us':>10}"
          f"{'之后占比':>10}{'vLLM us':>10}")
    for rows, vocab, top_k, _ in CASES:
        torch.manual_seed(42)
        logits = torch.randn((rows, vocab), dtype=torch.float32, device=DEV)
        starts = torch.zeros((rows,), dtype=torch.int32, device=DEV)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=DEV)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=DEV)
        steps = torch.zeros((rows,), dtype=torch.int32, device=DEV)
        nw = _num_warps(NUM_THREADS_PER_BLOCK)
        try:
            hist = triton.testing.do_bench(
                lambda: _count_steps[(rows,)](
                    logits, idx, starts, ends, steps,
                    logits.stride(0), logits.stride(1), vocab,
                    TOPK=top_k, TOPKP=triton.next_power_of_2(top_k),
                    BLOCK_SIZE=NUM_THREADS_PER_BLOCK, num_warps=nw,
                ), warmup=25, rep=100, return_mode="median")
            full = triton.testing.do_bench(
                lambda: flaggems_vllm.top_k_per_row_prefill(
                    logits, starts, ends, idx, rows,
                    logits.stride(0), logits.stride(1), top_k,
                ), warmup=25, rep=100, return_mode="median")
        except Exception as e:  # noqa: BLE001
            print(f"  {f'({rows},{vocab})':>16}   失败 {type(e).__name__}")
            continue
        v = float("nan")
        try:
            import vllm._custom_ops  # noqa: F401

            if hasattr(torch.ops._C, "top_k_per_row_prefill"):
                v = triton.testing.do_bench(
                    lambda: torch.ops._C.top_k_per_row_prefill(
                        logits, starts, ends, idx, rows,
                        logits.stride(0), logits.stride(1), top_k,
                    ), warmup=25, rep=100, return_mode="median") * 1000
        except Exception:  # noqa: BLE001
            pass
        after = (full - hist) * 1000
        print(f"  {f'({rows},{vocab})':>16}{hist*1000:>11.1f}{full*1000:>11.1f}"
              f"{after:>10.1f}{after/(full*1000)*100:>9.0f}%{v:>10.1f}", flush=True)
    print("\n  注意: rows <= SM 数时 wide-block 门控让算子用 BLOCK=1024 而探针是 512,")
    print("  那一行的差值无意义。只看 rows > SM 数的行。")

    # --- 完整分解：扫描 vs _process_bins vs final select，全部实测 ---
    print("\n" + "=" * 78)
    print("  完整分解 (只取门控不触发的形状, 全部实测)")
    print("=" * 78)
    print(f"  {'shape':>16}{'扫描':>9}{'process_bins':>14}{'final sel':>11}"
          f"{'整算子':>9}{'vLLM':>9}")
    for rows, vocab, top_k, _ in CASES:
        if rows <= _wide_block_max_rows():
            print(f"  {f'({rows},{vocab})':>16}   跳过: 门控触发, 探针与算子 BLOCK 不同")
            continue
        torch.manual_seed(42)
        logits = torch.randn((rows, vocab), dtype=torch.float32, device=DEV)
        starts = torch.zeros((rows,), dtype=torch.int32, device=DEV)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=DEV)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=DEV)
        steps = torch.zeros((rows,), dtype=torch.int32, device=DEV)
        sink = torch.empty((rows,), dtype=torch.int32, device=DEV)
        nw = _num_warps(NUM_THREADS_PER_BLOCK)
        try:
            scan = triton.testing.do_bench(
                lambda: _k_scan_only[(rows,)](
                    logits, sink, vocab, BLOCK=NUM_THREADS_PER_BLOCK,
                    NUM_BINS=2048, VEC=4, num_warps=nw,
                ), warmup=25, rep=100, return_mode="median") * 1000
            hist = triton.testing.do_bench(
                lambda: _count_steps[(rows,)](
                    logits, idx, starts, ends, steps,
                    logits.stride(0), logits.stride(1), vocab,
                    TOPK=top_k, TOPKP=triton.next_power_of_2(top_k),
                    BLOCK_SIZE=NUM_THREADS_PER_BLOCK, num_warps=nw,
                ), warmup=25, rep=100, return_mode="median") * 1000
            full = triton.testing.do_bench(
                lambda: flaggems_vllm.top_k_per_row_prefill(
                    logits, starts, ends, idx, rows,
                    logits.stride(0), logits.stride(1), top_k,
                ), warmup=25, rep=100, return_mode="median") * 1000
        except Exception as e:  # noqa: BLE001
            print(f"  {f'({rows},{vocab})':>16}   失败 {type(e).__name__}: {str(e)[:34]}")
            continue
        v = float("nan")
        try:
            import vllm._custom_ops  # noqa: F401

            if hasattr(torch.ops._C, "top_k_per_row_prefill"):
                v = triton.testing.do_bench(
                    lambda: torch.ops._C.top_k_per_row_prefill(
                        logits, starts, ends, idx, rows,
                        logits.stride(0), logits.stride(1), top_k,
                    ), warmup=25, rep=100, return_mode="median") * 1000
        except Exception:  # noqa: BLE001
            pass
        print(f"  {f'({rows},{vocab})':>16}{scan:>9.1f}{hist-scan:>14.1f}"
              f"{full-hist:>11.1f}{full:>9.1f}{v:>9.1f}", flush=True)
    print("\n  三段之和 = 整算子。最大的一段就是唯一值得重写的地方。")

    # --- Pass B 分层：它比 Pass A 慢 1.75x/字节, 慢在哪 ---
    print("\n" + "=" * 78)
    print("  Pass B 分层 (真 _process_bins + 真阈值 + 尾循环 + 防 DCE)")
    print("=" * 78)
    print(f"  {'shape':>16}{'真阈值':>8}{'触发率':>8}{'floor':>9}"
          f"{'+直写':>9}{'+final':>9}{'PassA':>9}")
    for rows, vocab, top_k, _ in CASES:
        if rows <= _wide_block_max_rows():
            continue
        torch.manual_seed(42)
        logits = torch.randn((rows, vocab), dtype=torch.float32, device=DEV)
        sink = torch.empty((rows,), dtype=torch.int32, device=DEV)
        nw = _num_warps(NUM_THREADS_PER_BLOCK)
        try:
            thrs = _real_thresholds(logits, top_k)
        except Exception as e:  # noqa: BLE001
            print(f"  {f'({rows},{vocab})':>16}  阈值计算失败 {type(e).__name__}")
            continue
        rate = top_k / vocab * 100
        r = {}
        try:
            r["floor"] = triton.testing.do_bench(
                lambda: _k_passB_floor[(rows,)](
                    logits, sink, vocab, BLOCK=NUM_THREADS_PER_BLOCK, VEC=4,
                    num_warps=nw,
                ), warmup=25, rep=100, return_mode="median") * 1000
            for tag, wd, uf in (("wd", True, False), ("both", True, True)):
                r[tag] = triton.testing.do_bench(
                    lambda w=wd, u=uf: _k_passB[(rows,)](
                        logits, sink, vocab, thrs,
                        BLOCK=NUM_THREADS_PER_BLOCK, NUM_BINS=2048, VEC=4,
                        WRITE_DIRECTLY=w, USE_FINAL=u, num_warps=nw,
                    ), warmup=25, rep=100, return_mode="median") * 1000
            pa = triton.testing.do_bench(
                lambda: _k_scan_only[(rows,)](
                    logits, sink, vocab, BLOCK=NUM_THREADS_PER_BLOCK,
                    NUM_BINS=2048, VEC=4, num_warps=nw,
                ), warmup=25, rep=100, return_mode="median") * 1000
        except Exception as e:  # noqa: BLE001
            print(f"  {f'({rows},{vocab})':>16}  失败 {type(e).__name__}: {str(e)[:32]}")
            continue
        print(f"  {f'({rows},{vocab})':>16}{int(thrs[0]):>8}{rate:>7.1f}%"
              f"{r['floor']:>9.1f}{r['wd']:>9.1f}{r['both']:>9.1f}{pa:>9.1f}",
              flush=True)
        # 聚合版：同样两条路径, 但每 tile 一次原子
        try:
            ag = triton.testing.do_bench(
                lambda: _k_passB_agg[(rows,)](
                    logits, sink, vocab, thrs,
                    BLOCK=NUM_THREADS_PER_BLOCK, VEC=4, USE_FINAL=True,
                    num_warps=nw,
                ), warmup=25, rep=100, return_mode="median") * 1000
            atom_now = r["both"] - r["floor"]
            atom_agg = ag - r["floor"]
            cut = (1 - atom_agg / atom_now) * 100 if atom_now > 0 else float("nan")
            newop = 217.5 - (atom_now - atom_agg) if rows == 64 else float("nan")
            print(f"      聚合版 {ag:>7.1f} us   原子成本 {atom_now:>6.1f} -> "
                  f"{atom_agg:>6.1f} us ({cut:>5.1f}% 削减)"
                  + (f"   算子 -> {newop:.1f}us, speedup {141.6/newop:.3f}"
                     if newop == newop else ""), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"      聚合版失败: {type(e).__name__}: {str(e)[:52]}")

    print("\n  floor 与 PassA 相近 => 重读本身不比建直方图贵, 差价在原子/store")
    print("  floor 就远超 PassA   => 贵在重读, 只能减少遍数(算法结构, override 改不动)")
    print("  健全性: floor 的带宽不得超过 1.3 TB/s, 超了就是又被 DCE 了")
    return 0


if __name__ == "__main__":
    sys.exit(main())
