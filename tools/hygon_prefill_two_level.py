"""Two-level threshold search: group totals first, then one group's bins.

WHY. The shipped step does ONE RADIX_SIZE-wide `tl.cumsum` per row to find the
bin holding rank top_k. Replacing the old carried chain of rounds with that
single scan was worth 0.658 -> 0.704, so the scan is not free. It is also the
only part of STEP 0 that is a full-width collective: the clear is a store, the
histogram is atomics, and the collect is a pass.

A two-level search reduces the 2048 bins to G group totals, cumsums those,
picks the group that straddles rank top_k, and then scans only that group's
2048/G bins. It reads the same histogram; what changes is the width of the
prefix sums, from one 2048-wide to one G-wide plus one (2048/G)-wide.

STEP 3 keeps the one-scan form, because it stores the full prefix back into
the histogram for the collect to use. Only STEP 0 changes -- which is the whole
of the algorithm on every benchmark shape, STEP 1-3 being dead code there.

WHY THE PROFILER IS ALLOWED HERE. Every arm makes exactly ONE launch, the same
kernel with the same grid, so nothing hides in the gaps between launches. That
is not true of the sampled path, where three launches against one made the
profiler overstate a win by 19% (tools/hygon_prefill_sample_bench.py), and it
is why that question had to go through the benchmark. This one does not.

ARMS: the shipped one-scan step, then G = 16, 32, 64 (groups of 128, 64, 32).

Each arm is its own patched copy of the generic module, and the dense arms get
the override's prefix-sum collection spliced in as SOURCE -- borrowing the
function object instead resolves its `_extract_bin_idx` in the override's
globals, which once produced wrong answers reported as a 1.40-1.45x speedup.

    tools/vendor_probe.sh tools/hygon_prefill_two_level.py hygon_prefill_two_level
"""

import importlib.util
import math
import pathlib
import subprocess
import sys
import tempfile
from importlib import import_module

import torch
from torch.profiler import ProfilerActivity, profile

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
_ov = import_module("flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill")

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
GROUPS = (16, 32, 64)


def occupancy(tag):
    import shutil

    for cmd in (["hy-smi"], ["rocm-smi"]):
        exe = shutil.which(cmd[0]) or (
            f"/opt/dtk/bin/{cmd[0]}"
            if pathlib.Path(f"/opt/dtk/bin/{cmd[0]}").exists()
            else None
        )
        if not exe:
            continue
        print(f"--- card occupancy {tag}: {cmd[0]}")
        out = subprocess.run([exe] + cmd[1:], capture_output=True, text=True).stdout
        print("\n".join(out.strip().splitlines()[:25]))
        return
    print(f"--- card occupancy {tag}: no smi tool found")


def two_level(groups):
    per = 2048 // groups
    return f"""    if STEP == 3:
        counts = tl.load(s_histogram_ptr + radix_bins)
        incl = last_value + tl.cumsum(counts, axis=0)
        prefix_sum = incl - counts
        threshold_mask = (prefix_sum < TOPK) & (incl >= TOPK)
        threshold_bin_idx = tl.min(
            tl.where(threshold_mask, radix_bins, RADIX_SIZE), axis=0
        ).to(tl.int32)
        final_bin_size = tl.max(tl.where(threshold_mask, counts, 0), axis=0)
        tl.store(s_histogram_ptr + radix_bins, prefix_sum)
        tl.debug_barrier()
    else:
        g_counts = tl.reshape(
            tl.load(s_histogram_ptr + radix_bins), ({groups}, {per})
        )
        g_tot = tl.sum(g_counts, axis=1)
        g_incl = last_value + tl.cumsum(g_tot, axis=0)
        g_pre = g_incl - g_tot
        g_hit = (g_pre < TOPK) & (g_incl >= TOPK)
        g_arr = tl.arange(0, {groups})
        g_idx = tl.min(tl.where(g_hit, g_arr, {groups}), axis=0).to(tl.int32)
        # a threshold always exists (rows no longer than top_k return earlier),
        # but clamp anyway so a miss reads this row rather than the next one's
        g_idx = tl.minimum(g_idx, {groups - 1})
        g_base = tl.max(tl.where(g_hit, g_pre, 0), axis=0)
        p_arr = tl.arange(0, {per})
        sub = tl.load(s_histogram_ptr + g_idx * {per} + p_arr)
        s_incl = g_base + tl.cumsum(sub, axis=0)
        s_pre = s_incl - sub
        s_hit = (s_pre < TOPK) & (s_incl >= TOPK)
        threshold_bin_idx = (
            g_idx * {per} + tl.min(tl.where(s_hit, p_arr, {per}), axis=0)
        ).to(tl.int32)
        final_bin_size = tl.max(tl.where(s_hit, sub, 0), axis=0)
"""


def slotscan_source():
    """The override's prefix-sum collection, sliced out of the override FILE."""
    ov_src = pathlib.Path(_ov.__file__).read_text()
    a = ov_src.index("@triton.jit\ndef _alloc_slots(")
    b = ov_src.index("# Rebind in the COPY only, before anything compiles.")
    return (
        "\n\n" + ov_src[a:b].rstrip() + "\n\n_process_bins = _process_bins_slotscan\n"
    )


def patched_source(groups, dense):
    """One-scan, then optionally the two-level scan over it."""
    src = pathlib.Path(_generic.__file__).read_text()
    for old, new in (
        (_ov._ONESCAN_CLEAR_OLD, _ov._ONESCAN_CLEAR_NEW),
        (_ov._ONESCAN_SCAN_OLD, _ov._ONESCAN_SCAN_NEW),
    ):
        n = src.count(old)
        assert n == 1, f"one-scan block found {n} times -- the operator changed"
        src = src.replace(old, new, 1)
    if groups is not None:
        n = src.count(_ov._ONESCAN_SCAN_NEW)
        assert n == 1, f"the one-scan block is present {n} times after patching"
        src = src.replace(_ov._ONESCAN_SCAN_NEW, two_level(groups), 1)
    return src + (slotscan_source() if dense else "")


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
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="twolevel_"))
    arms = [("base", None)] + [(f"g{g}", g) for g in GROUPS]
    mods = {}
    for tag, g in arms:
        for kind in ("generic", "dense"):
            f = tmp / f"tl_{tag}_{kind}.py"
            f.write_text(patched_source(g, kind == "dense"))
            mods[(tag, kind)] = load(f"flaggems_vllm.ops._tl_{tag}_{kind}", str(f))
    dev = "cuda"
    occupancy("before")
    print(
        f"\ndevice us, interleaved over {ROUNDS} rounds, each arm's FASTEST round."
        "\nOne launch per arm, same kernel and grid, so the profiler is sound here."
        "\nRatios are base/arm, so > 1 means the two-level search is faster.\n"
    )
    head = f"  {'shape':>18} {'module':>8}"
    for tag, _ in arms:
        head += f"{tag:>10}"
    for tag, _ in arms[1:]:
        head += f"{'b/' + tag:>9}"
    print(head + f"{'normal':>8}{'tied':>6}")

    logs = {tag: [] for tag, _ in arms[1:]}
    for rows, vocab, top_k, stride0 in SHAPES:
        kind = "dense" if vocab <= _ov.DENSE_VOCAB_PER_TOPK * top_k else "generic"
        geo = _ov._geometry(rows, vocab) if _ov._GEOMETRY else None
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        tbuf = (buf * 4).round() / 4
        tied = torch.as_strided(tbuf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)

        mins, oks = {}, {"normal": True, "tied": True}
        for tag, _ in arms:
            m = mods[(tag, kind)]
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
                good = torch.allclose(got, want) and bool((idx >= 0).all())
                if not good:
                    print(f"      ! {tag} WRONG on {label}", flush=True)
                oks[label] = oks[label] and good
            mins[tag] = min(
                device_us(lambda run=run: run(logits)) for _ in range(ROUNDS)
            )

        line = f"  {f'{rows}x{vocab}':>18} {kind:>8}"
        for tag, _ in arms:
            line += f"{mins[tag]:>10.1f}"
        for tag, _ in arms[1:]:
            r = mins["base"] / mins[tag]
            logs[tag].append(math.log(r))
            line += f"{r:>9.3f}"
        line += f"{'OK' if oks['normal'] else 'WRONG':>8}"
        line += f"{'OK' if oks['tied'] else 'WRONG':>6}"
        print(line, flush=True)

    print()
    for tag, _ in arms[1:]:
        print(f"  geomean base/{tag}: {math.exp(sum(logs[tag]) / len(logs[tag])):.3f}")
    occupancy("after")
    print(
        "\n  The scan is one collective among a store, atomics and a pass, so a"
        "\n  win here is bounded by whatever share of STEP 0 it holds; a loss"
        "\n  means the two narrower prefix sums cost more than the wide one."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
