"""What is a single pass over prefill's rows actually worth, in GB/s?

The sampled-threshold probe made ONE pass over (64,129280) in 200 us -- 165
GB/s -- while the shipped operator makes TWO passes plus everything else in
277 us (252 GB/s) and vLLM's whole op takes 108 us (306 GB/s). Our single pass
is slower per element than either competitor's entire algorithm, so the number
of passes may not be prefill's problem at all.

This measures the floor and then adds back the work, one layer at a time:

    read    load the row, accumulate -- no keys, no atomics, no writes
    key     + the operator's STEP-0 fp16 hi-11-bit key
    hist    + one atomic into a per-row 2048-bin histogram (the exact
            algorithm's first pass)
    append  + a threshold compare and a masked append into a candidate
            buffer through a shared counter (the sampled algorithm's pass)

Geometry is swept as a 2-D BLOCK x warps grid, not one axis at a time, and
the row split separately: on this card the best config has never been on
either axis alone. Reported in GB/s so the shapes are comparable, with the
shipped operator and vLLM alongside.

DEVICE time from the profiler. Our own Triton kernels' profiler time matches
the benchmark's do_bench exactly (1354 vs 1352 us and so on), but the vLLM
baseline's does NOT -- it comes out at 1.94-1.99x the benchmark's number on
five of seven shapes, with a ROCTracer "duplicate flow start" warning in the
log. So vLLM is timed here with CUDA events, the way do_bench does it, and
never with the profiler.

    tools/vendor_probe.sh tools/hygon_prefill_bandwidth.py hygon_prefill_bw
"""

import sys
from importlib import import_module

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

import flaggems_vllm

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
_key = _generic._convert_to_trt_uint16_hi11

SHAPES = [
    (64, 129280, 1024, 129280, 1),
    (4, 8193, 512, 8456, 1),
    (4, 16385, 512, 16648, 1),
    (16383, 4095, 512, 4352, 1),
    (12961, 4100, 512, 4360, 1),
    (16380, 5115, 512, 5376, 1),
    (4100, 1025, 512, 1288, 1),
]

# (BLOCK, num_warps) -- a grid, because the best point has never been on one
# axis alone on this card.
GEOMS = [
    (256, 2),
    (256, 4),
    (512, 4),
    (512, 8),
    (1024, 8),
    (1024, 16),
]
NB = 2048
CAP = 8192


def wall_us(fn, iters=30, warmup=10):
    """CUDA events around a loop, as do_bench does it."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000


def device_us(fn, iters=20, warmup=5):
    """Total Triton device time per call. Not valid for the vLLM op here."""
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


@triton.jit
def k_pass(
    logits_ptr,
    starts_ptr,
    ends_ptr,
    hist_ptr,
    cnt_ptr,
    cand_idx_ptr,
    cand_val_ptr,
    out_ptr,
    stride0,
    thr,
    LEVEL: tl.constexpr,
    CHUNK: tl.constexpr,
    SPLIT: tl.constexpr,
    CAP: tl.constexpr,
    NB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One pass over a row's chunk, doing LEVEL layers of the real work:
    0 read, 1 + key, 2 + histogram atomic, 3 + threshold compare and append."""
    pid = tl.program_id(0)
    row = pid // SPLIT
    chunk = pid % SPLIT
    lane = tl.arange(0, BLOCK)
    s = tl.load(starts_ptr + row)
    e = tl.load(ends_ptr + row)
    start = s + chunk * CHUNK
    end = tl.minimum(start + CHUNK, e)
    base = hist_ptr + row * NB
    cnt_ptrs = cnt_ptr + row + tl.zeros([BLOCK], tl.int32)
    acc = tl.zeros([BLOCK], tl.float32)
    kacc = tl.zeros([BLOCK], tl.int32)
    for t in tl.range(0, tl.cdiv(CHUNK, BLOCK)):
        i = start + t * BLOCK + lane
        m = i < end
        x = tl.load(logits_ptr + row * stride0 + i, mask=m, other=0.0)
        if LEVEL == 0:
            acc += x
        else:
            k = _key(x)
            if LEVEL == 1:
                kacc += k
            elif LEVEL == 2:
                tl.atomic_add(
                    base + k,
                    tl.full([BLOCK], 1, tl.int32),
                    mask=m,
                    sem="relaxed",
                    scope="cta",
                )
            else:
                take = m & (k <= thr)
                pos = tl.atomic_add(
                    cnt_ptrs,
                    tl.full([BLOCK], 1, tl.int32),
                    mask=take,
                    sem="relaxed",
                    scope="gpu",
                )
                keep = take & (pos < CAP)
                tl.store(
                    cand_idx_ptr + row * CAP + pos, (i - s).to(tl.int32), mask=keep
                )
                tl.store(cand_val_ptr + row * CAP + pos, x, mask=keep)
    if LEVEL <= 1:
        tl.store(
            out_ptr + pid, tl.sum(acc, axis=0) + tl.sum(kacc, axis=0).to(tl.float32)
        )


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    import vllm._custom_ops  # noqa: F401

    dev = "cuda"
    sms = ov._sm_count()
    levels = ("read", "key", "hist", "append")
    print(f"{sms} SMs; GB/s over the row bytes actually read, one pass\n")
    for rows, vocab, top_k, stride0, stride1 in SHAPES:
        torch.manual_seed(42)
        buf = torch.randn(
            (rows - 1) * stride0 + (vocab - 1) * stride1 + 1,
            device=dev,
            dtype=torch.float32,
        )
        logits = torch.as_strided(buf, (rows, vocab), (stride0, stride1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)
        hist = torch.zeros((rows, NB), dtype=torch.int32, device=dev)
        cnt = torch.zeros((rows,), dtype=torch.int32, device=dev)
        cand_idx = torch.empty((rows, CAP), dtype=torch.int32, device=dev)
        cand_val = torch.empty((rows, CAP), dtype=torch.float32, device=dev)
        gbytes = rows * vocab * 4 / 1e9

        t_vllm = wall_us(
            lambda: torch.ops._C.top_k_per_row_prefill(
                logits, starts, ends, idx, rows, stride0, stride1, top_k
            )
        )
        t_ship = device_us(
            lambda: flaggems_vllm.top_k_per_row_prefill(
                logits, starts, ends, idx, rows, stride0, stride1, top_k
            )
        )
        print(
            f"  {rows} x {vocab}, top_k {top_k}: {gbytes * 1e3:.1f} MB; "
            f"vLLM {t_vllm:.1f} us ({gbytes / t_vllm * 1e6:.0f} GB/s), "
            f"shipped {t_ship:.1f} us ({gbytes * 2 / t_ship * 1e6:.0f} GB/s "
            f"over two passes)"
        )
        header = f"    {'split':>5} {'B x w':>9}"
        for name in levels:
            header += f"{name:>10}"
        print(header)
        best = {}
        splits = [1]
        while rows * splits[-1] * 2 <= 8 * sms and vocab // (splits[-1] * 2) >= 512:
            splits.append(splits[-1] * 2)
        for split in splits:
            chunk = triton.cdiv(vocab, split)
            for block, warps in GEOMS:
                if block > chunk:
                    continue
                out = torch.empty((rows * split,), dtype=torch.float32, device=dev)
                cells = []
                for level in range(4):
                    launch = ov._Launch(
                        k_pass,
                        (rows * split,),
                        {
                            "LEVEL": level,
                            "CHUNK": chunk,
                            "SPLIT": split,
                            "CAP": CAP,
                            "NB": NB,
                            "BLOCK": block,
                        },
                        warps,
                    )

                    def run(launch=launch):
                        cnt.zero_()
                        launch(
                            logits,
                            starts,
                            ends,
                            hist,
                            cnt,
                            cand_idx,
                            cand_val,
                            out,
                            stride0,
                            1024,
                        )

                    t = device_us(run) - device_us(lambda: cnt.zero_())
                    cells.append(gbytes / max(t, 1e-6) * 1e6)
                    key = levels[level]
                    if cells[-1] > best.get(key, (0,))[0]:
                        best[key] = (cells[-1], split, block, warps)
                print(
                    f"    {split:>5} {f'{block} x {warps}':>9}"
                    + "".join(f"{c:>10.0f}" for c in cells)
                )
        print(
            "    best:",
            ", ".join(
                f"{k} {v[0]:.0f} GB/s at split {v[1]}, {v[2]}x{v[3]}"
                for k, v in best.items()
            ),
            "\n",
        )


if __name__ == "__main__":
    sys.exit(main())
