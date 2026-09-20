"""Measure Hygon costs of the primitive operations used by prefill.

This is not a replacement kernel.  It compares small isolated Triton kernels
for float16 conversion, cumsum, sum/reduce_or, and debug barriers.  The output
is deliberately reported as absolute device time plus a simple copy baseline;
it is a cost probe, not a claim that any primitive can be removed without
preserving the operator's algorithm.
"""

import json
import statistics
import sys

import torch
import triton
import triton.language as tl


ROWS = 4096
BLOCK = 2048
WARPS = (2, 4, 8)


@triton.jit
def _copy_f32(inp, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    x = tl.load(inp + pid * BLOCK + off)
    tl.store(out + pid * BLOCK + off, x)


@triton.jit
def _cvt_roundtrip(inp, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    x = tl.load(inp + pid * BLOCK + off)
    h = x.to(tl.float16)
    y = h.to(tl.float32)
    tl.store(out + pid * BLOCK + off, y)


@triton.jit
def _copy_i32(inp, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    x = tl.load(inp + pid * BLOCK + off)
    tl.store(out + pid * BLOCK + off, x)


@triton.jit
def _cumsum_i32(inp, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    x = tl.load(inp + pid * BLOCK + off)
    y = tl.cumsum(x, axis=0)
    tl.store(out + pid * BLOCK + off, y)


@triton.jit
def _sum_i32(inp, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    x = tl.load(inp + pid * BLOCK + off)
    y = tl.sum(x, axis=0)
    tl.store(out + pid, y)


@triton.jit
def _reduce_or(inp, out, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    x = tl.load(inp + pid * BLOCK + off).to(tl.int1)
    y = tl.reduce_or(x, axis=0)
    tl.store(out + pid, y.to(tl.int32))


@triton.jit
def _barrier(inp, out, BARRIERS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    x = tl.load(inp + pid * BLOCK + off)
    if BARRIERS >= 1:
        tl.debug_barrier()
    if BARRIERS >= 2:
        tl.debug_barrier()
    if BARRIERS >= 3:
        tl.debug_barrier()
    tl.store(out + pid * BLOCK + off, x)


def emit(kind, **fields):
    print(json.dumps(dict(kind=kind, **fields), sort_keys=True), flush=True)


def bench(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def main():
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.warp_size != 64:
        raise RuntimeError(f"Expected Hygon HIP wave64, got {target}")
    torch.manual_seed(7)
    f32 = torch.randn((ROWS, BLOCK), device="cuda", dtype=torch.float32)
    f32_out = torch.empty_like(f32)
    i32 = torch.arange(BLOCK, device="cuda", dtype=torch.int32).repeat(ROWS, 1)
    i32_out = torch.empty_like(i32)
    scalar = torch.empty((ROWS,), device="cuda", dtype=torch.int32)
    flags = (i32 & 31 == 0)
    flag_out = torch.empty((ROWS,), device="cuda", dtype=torch.int32)
    grid = (ROWS,)

    for warps in WARPS:
        def launch(kernel, *args, **meta):
            return kernel[grid](*args, num_warps=warps, **meta)

        cases = [
            ("copy_f32", lambda: launch(_copy_f32, f32, f32_out, BLOCK=BLOCK)),
            ("cvt_f32_f16_f32", lambda: launch(_cvt_roundtrip, f32, f32_out, BLOCK=BLOCK)),
            ("copy_i32", lambda: launch(_copy_i32, i32, i32_out, BLOCK=BLOCK)),
            ("cumsum_i32", lambda: launch(_cumsum_i32, i32, i32_out, BLOCK=BLOCK)),
            ("sum_i32", lambda: launch(_sum_i32, i32, scalar, BLOCK=BLOCK)),
            ("reduce_or", lambda: launch(_reduce_or, flags, flag_out, BLOCK=BLOCK)),
        ]
        for name, fn in cases:
            us = bench(fn)
            emit("primitive", op=name, warps=warps, us=us,
                 ns_per_element=us * 1000.0 / (ROWS * BLOCK if name not in ("sum_i32", "reduce_or") else ROWS))
        for count in (0, 1, 2, 3):
            fn = lambda count=count: launch(
                _barrier, f32, f32_out, BARRIERS=count, BLOCK=BLOCK
            )
            emit("primitive", op=f"barrier_{count}", warps=warps, us=bench(fn),
                 ns_per_element=None)
    emit("primitive_summary", rows=ROWS, block=BLOCK, warps=WARPS,
         note="Conversion result is round-tripped to fp32; compare against copy_f32.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
