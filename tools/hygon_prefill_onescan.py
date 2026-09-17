"""One threshold scan instead of a carried chain of rounds.

The operator's histogram step clears the 2048 (STEP 3: 1024) bins in
threshold_rounds = RADIX_SIZE // BLOCK_SIZE separate stores, and locates the
threshold in the same number of rounds, each a BLOCK_SIZE-wide cumsum whose
running total is carried into the next -- a serial chain -- plus two masked
stores of block-uniform scalars into global scratch per round, a barrier after
the loop, and two global loads to read those scalars back.

The DSA bin_topk kernel in this repo does the same job with one scan. So:

    clear      one vectorised store over the whole histogram
    threshold  one RADIX_SIZE-wide inclusive cumsum; the exclusive prefix is
               incl - counts, the bin is a single min-reduction and its size a
               single max-reduction -- both stay in registers, so the two
               global scalar stores, the barrier and the two reloads go away.
               Nothing downstream reads those two global buffers; the job
               takes threshold_bin_idx from the return value.

Nothing else changes. A threshold always exists here -- rows no longer than
top_k return before any histogram step -- so the reduction needs no
not-found case. STEP 3 writes its exclusive prefix over all 1024 bins where
the original wrote only the rounds up to the found one; STEP 1-3 never run on
the benchmark's inputs, and the rounded inputs below are there to make them
run anyway.

Why it may matter more than "the scan is in a 1.3 us base" suggested: the
many-row shapes run at BLOCK_SIZE 256, so their chain is EIGHT rounds deep,
and splitting T(top_k=1) at full and half range put their fixed cost at
148-155 us of 1067-1577. Why it may not: a 2048-wide cumsum on a 256- or
512-lane block needs a layout conversion that the rounds did not. Measured,
not assumed.

The patched operator is built by applying exactly two text replacements to
the generic module's CURRENT source on this machine, and the probe refuses to
run if either block is not found exactly once. Both arms are module copies at
the override's geometry, and the many-row shapes use the override's
prefix-sum _process_bins in both, as production does.

    tools/vendor_probe.sh tools/hygon_prefill_onescan.py hygon_prefill_onescan
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
ROUNDS = 3

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


def patched_source():
    src = pathlib.Path(_generic.__file__).read_text()
    for old, new in ((CLEAR_OLD, CLEAR_NEW), (SCAN_OLD, SCAN_NEW)):
        n = src.count(old)
        assert n == 1, f"expected block found {n} times -- the operator changed"
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
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="onescan_")) / "topk_prefill_onescan.py"
    tmp.write_text(patched_source())
    arms = {}
    for tag, path in (("rounds", _generic.__file__), ("onescan", str(tmp))):
        for kind in ("generic", "dense"):
            m = load(f"flaggems_vllm.ops._topk_prefill_{tag}_{kind}", path)
            if kind == "dense":
                m._process_bins = ov._process_bins_slotscan
            arms[(tag, kind)] = m
    dev = "cuda"
    print(
        "device us, today's rounds against one scan, interleaved; production"
        " routing and geometry in both arms\n"
    )
    print(
        f"  {'shape':>18} {'module':>8} {'rounds':>9} {'onescan':>9} "
        f"{'ratio':>7} {'spread':>13} {'normal':>7} {'tied':>6}"
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
        for tag in ("rounds", "onescan"):
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

        pairs = [(device_us(gos[0]), device_us(gos[1])) for _ in range(ROUNDS)]
        rs = sorted(a / b for a, b in pairs)
        med = rs[ROUNDS // 2]
        logs.append(math.log(med))
        a = sorted(p[0] for p in pairs)[ROUNDS // 2]
        b = sorted(p[1] for p in pairs)[ROUNDS // 2]
        print(
            f"  {f'{rows}x{vocab}':>18} {kind:>8} {a:>9.1f} {b:>9.1f} "
            f"{med:>7.3f} {rs[0]:>6.3f}-{rs[-1]:<6.3f} "
            f"{'OK' if oks['normal'] else 'WRONG':>7} "
            f"{'OK' if oks['tied'] else 'WRONG':>6}",
            flush=True,
        )
    print(f"\n  geomean ratio {math.exp(sum(logs) / len(logs)):.3f}")
    print(
        "  ratio > 1 means the single scan is faster. 'tied' uses rounded logits"
        "\n  so that the threshold bin overflows and STEP 1-3 actually run."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
