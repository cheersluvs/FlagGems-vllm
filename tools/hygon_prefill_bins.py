"""Fewer histogram bins: does the atomic get cheaper when it spreads less?

The STEP-0 probe established two things. STEP 0 always converges here -- the
threshold bin holds 27-36 elements at 2048 bins, so STEP 1-3 are dead code and
STEP 0 IS the algorithm. And 512 bins is safe: the bin grows to at most 868,
still inside NUM_FINAL_ITEMS = 2048, on every benchmark shape (256 bins is
not: 63 of 64 rows overflow on the long-row shape).

Safe is not the same as faster. Cutting bins does NOT reduce the number of
atomics -- still one per element -- it only reduces how many distinct
addresses they spread over, and on this card a same-address atomic costs 0.05
ns against 1.6 for a 2048-bin scatter. A replica kernel measured 256 bins at
1.4x of 2048 on the dense shapes, but replicas have already been wrong about
absolute rates here, so it has to be tried on the operator.

The bin count is not reachable by rebinding: RADIX11_SIZE and NUM_BINS are
constexprs declared INSIDE the kernels, so changing them means patching the
generic module's source, which breaks the moment upstream edits that file.

_extract_bin_idx is module level, though, and already rebound by the shipped
override. Shifting its STEP-0 key down by two more bits gives 512 distinct
bins while the histogram stays 2048 wide -- so the clear and the scan cost
exactly what they cost today and ONLY the address spread changes. That is the
one variable in doubt: the clear measured 0.47 us across four steps and the
scan lives inside a 1.3 us base, neither worth a source patch, while the
histogram pass itself is the dominant term.

If the spread alone pays, the fuller change earns its risk. If not, this is
the end of the bin-count idea and nothing was spent on it.

    tools/vendor_probe.sh tools/hygon_prefill_bins.py hygon_prefill_bins
"""

import importlib.util
import sys
from importlib import import_module

import torch
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
_convert_to_uint32 = _generic._convert_to_uint32

SHAPES = [
    (64, 129280, 1024, 129280),
    (4, 8193, 512, 8456),
    (4, 16385, 512, 16648),
    (16383, 4095, 512, 4352),
    (12961, 4100, 512, 4360),
    (16380, 5115, 512, 5376),
    (4100, 1025, 512, 1288),
]
# (label, replacement for _extract_bin_idx); None keeps the operator's own
VARIANTS = [
    ("2048 bins", None),
    ("1024 bins", "_extract_6"),
    ("512 bins", "_extract_7"),
]
ROUNDS = 3


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


def load_copy(name):
    """Another instance of the generic module, with its own jit globals."""
    spec = importlib.util.spec_from_file_location(name, _generic.__file__)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@triton.jit
def _map16(x):
    """The operator's own fp16 ordering, before the key is narrowed."""
    h = x.to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    sign_set = (bits & tl.full(bits.shape, 0x8000, tl.uint16)) != 0
    inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
    return tl.where(sign_set, bits, inv)


@triton.jit
def _refine(x, in_range, pattern, STEP: tl.constexpr):
    """STEP 1-3, verbatim from the operator. Dead code on these shapes -- STEP
    0 always converges -- but kept so the rebind is a drop-in."""
    bits = _convert_to_uint32(x)
    if STEP == 1:
        bin_idx = (bits >> 21) & 0x7FF
        is_partial_match = in_range & ((bits >> 21) == pattern)
    elif STEP == 2:
        bin_idx = (bits >> 10) & 0x7FF
        is_partial_match = in_range & ((bits >> 10) == pattern)
    else:
        bin_idx = bits & 0x3FF
        is_partial_match = in_range & (bits == pattern)
    return bin_idx, is_partial_match


# One function per variant with the shift written out. A closure variable is
# NOT usable here: Triton rejects any global or captured name inside a @jit
# function unless it is a tl.constexpr instance, which is what round 1 of this
# probe died on -- the third time this session (after BIG and SAFETY).
@triton.jit
def _extract_5(x, in_range, pattern, STEP: tl.constexpr):
    if STEP == 0:
        return (_map16(x) >> 5).to(tl.uint32), in_range
    return _refine(x, in_range, pattern, STEP)


@triton.jit
def _extract_6(x, in_range, pattern, STEP: tl.constexpr):
    if STEP == 0:
        return (_map16(x) >> 6).to(tl.uint32), in_range
    return _refine(x, in_range, pattern, STEP)


@triton.jit
def _extract_7(x, in_range, pattern, STEP: tl.constexpr):
    if STEP == 0:
        return (_map16(x) >> 7).to(tl.uint32), in_range
    return _refine(x, in_range, pattern, STEP)


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    dev = "cuda"
    mods = []
    for i, (label, repl) in enumerate(VARIANTS):
        mod = load_copy(f"flaggems_vllm.ops._topk_prefill_bins_{i}")
        if repl:
            mod._extract_bin_idx = globals()[repl]
        mods.append((label, mod))
    print("device us per variant, interleaved; only the STEP-0 key changes\n")
    print(
        f"  {'shape':>18} "
        + "".join(f"{lbl:>12}" for lbl, _ in mods)
        + f"{'best vs 2048':>14} {'ans':>5}"
    )
    for rows, vocab, top_k, stride0 in SHAPES:
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)
        want = torch.topk(logits, top_k, dim=1).values.sort(dim=1).values
        geo = ov._geometry(rows, vocab) if ov._GEOMETRY else None

        calls = []
        for label, mod in mods:
            if geo is None:
                mod.NUM_THREADS_PER_BLOCK, mod._num_warps = (
                    _generic.NUM_THREADS_PER_BLOCK,
                    _generic._num_warps,
                )
            else:
                mod.NUM_THREADS_PER_BLOCK = geo[0]
                mod._num_warps = lambda b, w=geo[1]: w

            def go(mod=mod):
                mod.top_k_per_row_prefill(
                    logits, starts, ends, idx, rows, stride0, 1, top_k
                )

            calls.append(go)

        ok = True
        for go in calls:
            idx.fill_(-9)
            go()
            torch.cuda.synchronize()
            got = logits.gather(1, idx.long().clamp(0, vocab - 1)).sort(dim=1).values
            ok = ok and torch.allclose(got, want) and bool((idx >= 0).all())

        sums = [0.0] * len(calls)
        for _ in range(ROUNDS):
            for i, go in enumerate(calls):
                sums[i] += device_us(go)
        ts = [t / ROUNDS for t in sums]
        print(
            f"  {f'{rows}x{vocab}':>18} "
            + "".join(f"{t:>12.1f}" for t in ts)
            + f"{ts[0] / min(ts):>14.3f} {'OK' if ok else 'WRONG':>5}",
            flush=True,
        )
    print(
        "\n  Only the STEP-0 key's width changes: the histogram is still 2048"
        "\n  wide, so the clear and the scan are identical across columns and"
        "\n  the difference is purely how far the atomics spread."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
