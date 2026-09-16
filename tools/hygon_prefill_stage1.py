"""Why is the same prefill kernel 1.45x faster inside my pipeline?

On (64,129280) a four-launch split-1 pipeline came out at 249 us of device
time against the shipped operator's 310 -- and its stage 1 is the SAME generic
kernel at the SAME geometry, since _geometry() returns None below four rows
per SM and the override just forwards the call. Subtracting the pipeline's
tail (~31 us at 64 rows and 1024 candidates) and gather leaves stage 1 at
roughly 214 us against 310.

Only two things differ, so separate them:

    shipped    the override, then the generic host dispatch, then the kernel
    generic    the generic op called directly, bypassing the override
    launch     the kernel launched directly, real stride0 and the caller's
               row bounds
    folded     the kernel launched directly with stride0 = 0 and the row
               offset folded into the bounds -- what the split pipeline does

(a) vs (b) prices the override, (b) vs (c) the launch mechanism, (c) vs (d)
the folded-offset form. Over a geometry grid, because the geometry sweep that
set today's rule measured this shape at only three points.

Device time from the profiler, which is what the benchmark's kernel mode
reports: do_bench brackets each call with events, so it measures per-call
device elapsed, not the host serialisation between calls. (Wall-clock loops
over these shapes read 5-10x higher and are not comparable to the benchmark.)

    tools/vendor_probe.sh tools/hygon_prefill_stage1.py hygon_prefill_stage1
"""

import sys
from importlib import import_module

import torch
from torch.profiler import ProfilerActivity, profile

import flaggems_vllm

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")

SHAPES = [
    (64, 129280, 1024, 129280, 1),
    (4, 8193, 512, 8456, 1),
    (4, 16385, 512, 16648, 1),
    (4100, 1025, 512, 1288, 1),
    (16383, 4095, 512, 4352, 1),
]
GEOMS = [(256, 2), (256, 4), (512, 4), (512, 8), (1024, 8), (1024, 16)]


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


def main():
    gen = _generic
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    import vllm._custom_ops  # noqa: F401

    dev = "cuda"
    print("device us; every number is the same kernel doing the same work\n")
    for rows, vocab, top_k, stride0, stride1 in SHAPES:
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        fstart = (
            torch.arange(rows, dtype=torch.int32, device=dev) * stride0
        ).contiguous()
        fend = (fstart + vocab).contiguous()
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)
        want = torch.topk(logits, top_k, dim=1).values.sort(dim=1).values
        scratch = (
            torch.empty((rows, gen.NUM_BINS), dtype=torch.int32, device=dev),
            torch.empty((rows, gen.NUM_FILNAL_ITEMS), dtype=torch.float32, device=dev),
            torch.empty((rows,), dtype=torch.int32, device=dev),
            torch.empty((rows,), dtype=torch.int32, device=dev),
            torch.empty((rows,), dtype=torch.int32, device=dev),
            torch.empty((rows,), dtype=torch.int32, device=dev),
        )

        t_vllm = device_us(
            lambda: torch.ops._C.top_k_per_row_prefill(
                logits, starts, ends, idx, rows, stride0, stride1, top_k
            )
        )
        t_ship = device_us(
            lambda: flaggems_vllm.top_k_per_row_prefill(
                logits, starts, ends, idx, rows, stride0, stride1, top_k
            )
        )
        t_gen = device_us(
            lambda: gen.top_k_per_row_prefill(
                logits, starts, ends, idx, rows, stride0, stride1, top_k
            )
        )
        print(
            f"  {rows} x {vocab}, top_k {top_k}: vLLM {t_vllm:.1f}, "
            f"shipped {t_ship:.1f}, generic op {t_gen:.1f}"
        )
        print(
            f"    {'B x w':>9} {'launch':>9} {'folded':>9} {'ans':>6} "
            f"{'best vs shipped':>16}"
        )
        for block, warps in GEOMS:
            cells = []
            for folded in (False, True):
                lj = ov._Launch(
                    gen.non_tle_top_k_per_row_prefill,
                    (rows,),
                    {"TOPK": top_k, "BLOCK_SIZE": block, "ROW_OFFSET": 0},
                    warps,
                )
                s0 = 0 if folded else stride0
                a = fstart if folded else starts
                b = fend if folded else ends

                def run(lj=lj, s0=s0, a=a, b=b):
                    lj(logits, idx, a, b, s0, 1, vocab, *scratch)

                idx.fill_(-9)
                run()
                torch.cuda.synchronize()
                got = (
                    logits.gather(1, idx.long().clamp(0, vocab - 1)).sort(dim=1).values
                )
                ok = torch.allclose(got, want) and bool((idx >= 0).all())
                cells.append((device_us(run), ok))
            best = min(c[0] for c in cells)
            print(
                f"    {f'{block} x {warps}':>9} {cells[0][0]:>9.1f} "
                f"{cells[1][0]:>9.1f} "
                f"{'OK' if cells[0][1] and cells[1][1] else 'WRONG':>6} "
                f"{t_ship / best:>16.2f}"
            )
        print()


if __name__ == "__main__":
    sys.exit(main())
