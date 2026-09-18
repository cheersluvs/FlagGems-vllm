"""Narrow STEP 0's histogram in the OPERATOR, not in a replica.

tools/hygon_histogram_cost2 measured, per element, on the operator's own tile
loop:

                  2048 bins   512    256
    atomic          0.605     0.458  0.418
    tl.histogram    5.226     0.564  0.268

Two things follow, and they need separating on the operator before either is
built.

1. The atomic ALONE gets cheaper as the bins narrow -- 0.605 -> 0.458 at 512.
   That contradicts what the operator said when I narrowed only the KEY and
   left the buffer and the scan 2048 wide (2048 best on six of seven shapes).
   A replica and the operator have already disagreed three times here, so the
   operator decides.
2. tl.histogram only beats the atomic at 256 bins (1.54x with a register
   accumulator; the per-tile vector atomic that _hygon/ops/persistent_topk.py
   uses is worse, 1.26x). But 256 bins overflows NUM_FINAL_ITEMS on 63 of 64
   rows of (64,129280) (tools/hygon_prefill_step0.py), forcing STEP 1 to run.
   It is safe on the many-row shapes, whose worst bin there was 269-322.

So this measures (1) first, because it is a small patch and it gates (2): if
narrowing the whole histogram does not pay on the operator, the 256-bin
tl.histogram path has nothing to stand on.

The arms are the SHIPPED one-scan step at 2048 STEP-0 bins against the same
step at 512 and at 256, applied to the generic source as text replacements on
top of the override's own patch, so the baseline is what production runs. The
sparse shapes are expected to lose at 256 -- that is the extra STEP 1 -- and
are measured anyway, since a shape-dependent bin count would need the number.

    tools/vendor_probe.sh tools/hygon_prefill_bins256.py hygon_prefill_bins256
"""

import importlib.util
import math
import pathlib
import sys
import tempfile
from importlib import import_module

import torch
from torch.profiler import ProfilerActivity, profile

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")

SHAPES = [
    (64, 129280, 1024, 129280),
    (4, 16385, 512, 16648),
    (4, 8193, 512, 8456),
    (16383, 4095, 512, 4352),
    (12961, 4100, 512, 4360),
    (16380, 5115, 512, 5376),
    (4100, 1025, 512, 1288),
]
ROUNDS = 7


def occupancy(tag):
    """What else is on the card. This box is shared, and round 1 of this probe
    came back with per-round ratios spread 0.08-4.2 while the same shapes had
    held within 10% that morning."""
    import shutil
    import subprocess

    for cmd in (["hy-smi"], ["rocm-smi", "--showpids"], ["rocm-smi"]):
        exe = shutil.which(cmd[0]) or (
            f"/opt/dtk/bin/{cmd[0]}"
            if pathlib.Path(f"/opt/dtk/bin/{cmd[0]}").exists()
            else None
        )
        if not exe:
            continue
        try:
            out = subprocess.run(
                [exe] + cmd[1:], capture_output=True, text=True, timeout=30
            ).stdout
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            out = repr(exc)
        print(f"--- card occupancy {tag}: {' '.join(cmd)}")
        print("\n".join(out.strip().splitlines()[:25]))
        return
    print(f"--- card occupancy {tag}: no smi tool found")


STEP0_BINS = (2048, 512, 256)

CLEAR_OLD = """    threshold_rounds: tl.constexpr = (
        RADIX10_SIZE // BLOCK_SIZE if STEP == 3 else RADIX11_SIZE // BLOCK_SIZE
    )
    for clear_round in tl.static_range(0, threshold_rounds):
        clear_bins = clear_round * BLOCK_SIZE + lane
        tl.store(s_histogram_ptr + clear_bins, 0)
    tl.debug_barrier()
"""
CLEAR_NEW = """    RADIX_SIZE: tl.constexpr = RADIX10_SIZE if STEP == 3 else RADIX11_SIZE
    radix_bins = tl.arange(0, RADIX_SIZE)
    tl.store(s_histogram_ptr + radix_bins, tl.zeros([RADIX_SIZE], tl.int32))
    tl.debug_barrier()
"""
SCAN_OLD = """    threshold_bin_ptrs = s_threshold_bin_idx_ptr + zeros
    final_bin_size_ptrs = s_final_bin_size_ptr + zeros
    threshold_found = tl.full((), False, dtype=tl.int1)
    for round_idx in tl.static_range(0, threshold_rounds):
        if not threshold_found:
            bins = round_idx * BLOCK_SIZE + lane
            counts = tl.load(s_histogram_ptr + bins)
            if HAS_TLE:
                prefix_sum, counts_total = tle.cumsum(counts, axis=0, reverse=False)
            else:
                counts_total = tl.sum(counts)
                prefix_sum = counts_total - tl.cumsum(counts, axis=0, reverse=True)
            prefix_sum = prefix_sum + last_value
            total_sum = last_value + counts_total
            next_prefix_sum = prefix_sum + counts
            threshold_mask = (prefix_sum < TOPK) & (next_prefix_sum >= TOPK)
            threshold_bin = bins
            threshold_bin_size = next_prefix_sum - prefix_sum
            if STEP == 3:
                tl.store(s_histogram_ptr + bins, prefix_sum)
            tl.store(threshold_bin_ptrs, threshold_bin, mask=threshold_mask)
            tl.store(final_bin_size_ptrs, threshold_bin_size, mask=threshold_mask)
            found_round = tl.reduce_or(threshold_mask, axis=0)
            threshold_found = found_round
            last_value = total_sum

    tl.debug_barrier()
    threshold_bin_idx = tl.load(s_threshold_bin_idx_ptr)
    final_bin_size = tl.load(s_final_bin_size_ptr)
"""
SCAN_NEW = """    counts = tl.load(s_histogram_ptr + radix_bins)
    incl = last_value + tl.cumsum(counts, axis=0)
    prefix_sum = incl - counts
    threshold_mask = (prefix_sum < TOPK) & (incl >= TOPK)
    threshold_bin_idx = tl.min(
        tl.where(threshold_mask, radix_bins, RADIX_SIZE), axis=0
    ).to(tl.int32)
    final_bin_size = tl.max(tl.where(threshold_mask, counts, 0), axis=0)
    if STEP == 3:
        tl.store(s_histogram_ptr + radix_bins, prefix_sum)
        tl.debug_barrier()
"""


RADIX_OLD = (
    "    RADIX_SIZE: tl.constexpr = RADIX10_SIZE if STEP == 3 else RADIX11_SIZE\n"
)
KEY_OLD = "        bin_idx = (mapped >> 5).to(tl.uint32)\n"


def patched_source(step0_bins):
    """The override's one-scan patch, then STEP 0 narrowed to step0_bins."""
    src = pathlib.Path(_generic.__file__).read_text()
    for old, new in ((CLEAR_OLD, CLEAR_NEW), (SCAN_OLD, SCAN_NEW)):
        n = src.count(old)
        assert n == 1, f"expected block found {n} times -- the operator changed"
        src = src.replace(old, new, 1)
    if step0_bins == 2048:
        return src
    shift = 5 + (11 - step0_bins.bit_length() + 1)
    for old, new in (
        (
            RADIX_OLD,
            "    RADIX_SIZE: tl.constexpr = (\n"
            "        RADIX10_SIZE\n"
            "        if STEP == 3\n"
            f"        else ({step0_bins} if STEP == 0 else RADIX11_SIZE)\n"
            "    )\n",
        ),
        (KEY_OLD, f"        bin_idx = (mapped >> {shift}).to(tl.uint32)\n"),
    ):
        n = src.count(old)
        assert n == 1, f"narrowing block found {n} times"
        src = src.replace(old, new, 1)
    return src


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="bins256_"))
    paths = []
    for nb in STEP0_BINS:
        f = tmpdir / f"topk_prefill_b{nb}.py"
        f.write_text(patched_source(nb))
        paths.append((f"b{nb}", str(f)))
    arms = {}
    for tag, path in paths:
        for kind in ("generic", "dense"):
            m = load(f"flaggems_vllm.ops._topk_prefill_{tag}_{kind}", path)
            if kind == "dense":
                m._process_bins = ov._process_bins_slotscan
            arms[(tag, kind)] = m
    dev = "cuda"
    occupancy("before")
    print(
        "\ndevice us, today's rounds against one scan, interleaved over"
        f" {ROUNDS} rounds; production routing and geometry in both arms."
        "\n'min' is each arm's least-disturbed round -- contention only ever"
        " ADDS time -- and is the number to read when the spread is wide.\n"
    )
    print(
        f"  {'shape':>18} {'module':>8}"
        + "".join(f"{'min ' + t:>10}" for t, _ in paths)
        + f"{'512/2048':>10} {'256/2048':>10} {'normal':>7} {'tied':>6}"
    )
    logs = []
    for rows, vocab, top_k, stride0 in SHAPES:
        kind = "dense" if vocab <= ov.DENSE_VOCAB_PER_TOPK * top_k else "generic"
        geo = ov._geometry(rows, vocab) if ov._GEOMETRY else None
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        tbuf = (buf * 4).round() / 4
        tied = torch.as_strided(tbuf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)

        gos = []
        oks = {"normal": True, "tied": True}
        for tag, _ in paths:
            m = arms[(tag, kind)]
            if geo is None:
                m.NUM_THREADS_PER_BLOCK = _generic.NUM_THREADS_PER_BLOCK
                m._num_warps = _generic._num_warps
            else:
                m.NUM_THREADS_PER_BLOCK = geo[0]
                m._num_warps = lambda b, w=geo[1]: w

            def run(src, m=m):
                m.top_k_per_row_prefill(src, starts, ends, idx, rows, stride0, 1, top_k)

            for label, src in (("normal", logits), ("tied", tied)):
                want = torch.topk(src, top_k, dim=1).values.sort(dim=1).values
                idx.fill_(-9)
                run(src)
                torch.cuda.synchronize()
                got = src.gather(1, idx.long().clamp(0, vocab - 1)).sort(dim=1).values
                oks[label] = (
                    oks[label] and torch.allclose(got, want) and bool((idx >= 0).all())
                )
            gos.append(lambda run=run: run(logits))

        mins = [min(device_us(g) for _ in range(ROUNDS)) for g in gos]
        logs.append(math.log(mins[0] / mins[-1]))
        print(
            f"  {f'{rows}x{vocab}':>18} {kind:>8}"
            + "".join(f"{m:>10.1f}" for m in mins)
            + f"{mins[0] / mins[1]:>10.3f} {mins[0] / mins[2]:>10.3f} "
            f"{'OK' if oks['normal'] else 'WRONG':>7} "
            f"{'OK' if oks['tied'] else 'WRONG':>6}",
            flush=True,
        )
    print(f"\n  geomean 2048/256 {math.exp(sum(logs) / len(logs)):.3f}")
    occupancy("after")
    print(
        "  ratio > 1 means the narrower STEP 0 is faster. 'tied' uses rounded"
        "\n  logits, which overflow the threshold bin and make STEP 1-3 run --"
        "\n  the same thing 256 bins is expected to do to the sparse shapes."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
