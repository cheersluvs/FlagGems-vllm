"""Fair benchmark for the split=1 pipeline, including scratch allocation.

The earlier split probe preallocated all candidate and radix scratch buffers,
while the shipped prefill wrapper allocates its non-TLE scratch on every call.
That comparison is useful as a kernel lower bound but not as an end-to-end
benchmark.  This probe measures three arms with the same external input and
output buffers:

* ``baseline``: current Hygon shipped operator, including its allocations;
* ``split1_prealloc``: split=1 pipeline with scratch allocated once;
* ``split1_alloc``: split=1 pipeline allocating every scratch tensor per call.

The last arm is the fair comparison requested for the possible alternate
single-CTA pipeline.  The probe uses event-bracketed wall time as the primary
metric, because that includes allocation and launch overhead.  A device-time
summary is also printed to separate kernel cost from host-side overhead.

Run on BW1000 with:

    tools/vendor_probe.sh tools/hygon_prefill_split_alloc_fair.py \
        hygon_prefill_split_alloc_fair_v1
"""

from __future__ import annotations

import statistics
import sys
from importlib import import_module
from pathlib import Path

import torch
import triton
from torch.profiler import ProfilerActivity, profile


TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import hygon_prefill_split_workset as split_probe  # noqa: E402


GENERIC = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
OV = import_module(
    "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
)

SHAPES = (
    (64, 129280, 1024, 129280),
    (4, 8193, 512, 8456),
    (4, 16385, 512, 16648),
    (16383, 4095, 512, 4352),
)
NB = 2048


def event_us(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iters


def device_us(fn, iters=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total = 0.0
    for ev in prof.key_averages():
        value = getattr(ev, "self_device_time_total", None)
        if value is None:
            value = getattr(ev, "self_cuda_time_total", 0.0)
        total += value or 0.0
    return total / iters


def split1_alloc_run(logits, starts, ends, out, rows, vocab, top_k, stride0):
    """Return a split=1 run whose complete scratch is allocated per call."""
    split = 1
    chunk = vocab
    nv = rows
    ncand = top_k
    cap = triton.next_power_of_2(ncand)
    block = GENERIC.NUM_THREADS_PER_BLOCK
    gblock = min(1024, cap)

    bounds = OV._Launch(
        split_probe._bounds,
        (rows,),
        {"SPLIT": 1, "CHUNK": chunk, "BLOCK": 1},
        1,
    )
    local_topk = OV._Launch(
        GENERIC.non_tle_top_k_per_row_prefill,
        (nv,),
        {"TOPK": top_k, "BLOCK_SIZE": block, "ROW_OFFSET": 0},
        GENERIC._num_warps(block),
    )
    gather = OV._Launch(
        split_probe._gather,
        (rows, triton.cdiv(ncand, gblock)),
        {"SPLIT": 1, "TOPK": top_k, "CHUNK": chunk, "NCAND": ncand, "BLOCK": gblock},
        4,
    )
    tail = OV._Launch(
        OV._tail,
        (rows,),
        {
            "TOPK": top_k,
            "NB": NB,
            "CAP": cap,
            "RADIX": split_probe.RADIX,
            "BLOCK": 512,
        },
        8,
    )
    floor = torch.finfo(torch.float32).min

    def run():
        # Deliberately keep these allocations inside the timed call.  This is
        # the same lifetime model as top_k_per_row_prefill's non-TLE wrapper.
        cstart = torch.empty((nv,), dtype=torch.int32, device="cuda")
        cend = torch.empty((nv,), dtype=torch.int32, device="cuda")
        cand = torch.empty((nv, top_k), dtype=torch.int32, device="cuda")
        cand_val = torch.empty((rows, cap), dtype=torch.float32, device="cuda")
        cand_idx = torch.empty((rows, cap), dtype=torch.int32, device="cuda")
        cnt = torch.full((rows,), ncand, dtype=torch.int32, device="cuda")
        hist = torch.empty((rows, NB), dtype=torch.int32, device="cuda")
        counts = torch.empty((rows, split_probe.RADIX), dtype=torch.int32, device="cuda")
        slot = torch.empty((rows,), dtype=torch.int32, device="cuda")
        scratch = (
            torch.empty((nv, GENERIC.NUM_BINS), dtype=torch.int32, device="cuda"),
            torch.empty(
                (nv, GENERIC.NUM_FILNAL_ITEMS), dtype=torch.float32, device="cuda"
            ),
            torch.empty((nv,), dtype=torch.int32, device="cuda"),
            torch.empty((nv,), dtype=torch.int32, device="cuda"),
            torch.empty((nv,), dtype=torch.int32, device="cuda"),
            torch.empty((nv,), dtype=torch.int32, device="cuda"),
        )
        bounds(starts, ends, cstart, cend, stride0)
        local_topk(logits, cand, cstart, cend, 0, 1, chunk, *scratch)
        gather(logits, starts, cand, cand_val, cand_idx, stride0, floor)
        tail(
            logits,
            ends,
            hist,
            cnt,
            cand_idx,
            cand_val,
            out,
            counts,
            slot,
            stride0,
        )

    return run


def validate(fn, out, logits, top_k, want):
    out.fill_(-9)
    fn()
    torch.cuda.synchronize()
    idx = out.long().clamp(0, logits.shape[1] - 1)
    got = logits.gather(1, idx).sort(dim=1).values
    if not bool(torch.allclose(got, want) and bool((out >= 0).all())):
        raise AssertionError("split1 arm failed exact top-k validation")


def readings(arms, rounds=4):
    values = {name: [] for name in arms}
    names = list(arms)
    for round_id in range(rounds):
        order = names[round_id % len(names) :] + names[: round_id % len(names)]
        if round_id % 2:
            order.reverse()
        for name in order:
            values[name].append(event_us(arms[name]))
    return values


def report_shape(rows, vocab, top_k, stride0, seed):
    torch.manual_seed(seed)
    buf = torch.randn(
        (rows - 1) * stride0 + vocab, device="cuda", dtype=torch.float32
    )
    logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
    starts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    ends = torch.full((rows,), vocab, dtype=torch.int32, device="cuda")
    want = torch.topk(logits, top_k, dim=1).values.sort(dim=1).values
    outputs = {
        "baseline": torch.empty((rows, top_k), dtype=torch.int32, device="cuda"),
        "split1_prealloc": torch.empty(
            (rows, top_k), dtype=torch.int32, device="cuda"
        ),
        "split1_alloc": torch.empty((rows, top_k), dtype=torch.int32, device="cuda"),
    }

    import flaggems_vllm

    def baseline():
        flaggems_vllm.top_k_per_row_prefill(
            logits, starts, ends, outputs["baseline"], rows, stride0, 1, top_k
        )

    prealloc, prealloc_out = split_probe.run_chunk_split(
        logits, starts, ends, rows, vocab, top_k, stride0, 1
    )

    def split1_prealloc():
        prealloc()

    alloc = split1_alloc_run(
        logits, starts, ends, outputs["split1_alloc"], rows, vocab, top_k, stride0
    )

    validate(baseline, outputs["baseline"], logits, top_k, want)
    validate(split1_prealloc, prealloc_out, logits, top_k, want)
    validate(alloc, outputs["split1_alloc"], logits, top_k, want)

    arms = {
        "baseline": baseline,
        "split1_prealloc": split1_prealloc,
        "split1_alloc": alloc,
    }
    wall = readings(arms)
    device = {name: device_us(fn) for name, fn in arms.items()}
    med = {name: statistics.median(vals) for name, vals in wall.items()}
    print(
        f"shape={rows}x{vocab} k={top_k} seed={seed} "
        f"wall_samples={wall}",
        flush=True,
    )
    print(
        "  wall_median_us "
        + " ".join(f"{name}={med[name]:.3f}" for name in arms),
        flush=True,
    )
    print(
        "  wall_ratio_vs_baseline "
        + " ".join(
            f"{name}={med['baseline'] / med[name]:.3f}"
            for name in ("split1_prealloc", "split1_alloc")
        ),
        flush=True,
    )
    print(
        f"  allocation_overhead split1_alloc/prealloc="
        f"{med['split1_alloc'] / med['split1_prealloc']:.3f}",
        flush=True,
    )
    print(
        "  device_us " + " ".join(f"{name}={device[name]:.3f}" for name in arms),
        flush=True,
    )


def main():
    import vllm._custom_ops  # noqa: F401

    print("## fair split=1 benchmark: allocation included", flush=True)
    for seed in (42, 43):
        for shape in SHAPES:
            report_shape(*shape, seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
