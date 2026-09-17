"""Clamp the STEP-0 key: can most histogram atomics share ONE address?

On this card an atomic whose lanes all hit one address costs ~0.05 ns, and a
2048-bin scatter ~1.6 ns -- 32x. The operator's histogram pass is a scatter,
one atomic per element, and it is about 103 us of the 230 at (64,129280).

But most of those elements do not need a bin. The pass only has to be exact
BELOW the threshold bin; everything above it merely has to be counted. So

    bin = min(key, CLAMP)        CLAMP a little above the threshold bin

keeps every bin that matters exact and folds the bulk into one overflow bin.
For the benchmark's standard-normal logits the threshold keys sit around
505-539, and a clamp of 520-560 folds ~98% of a sparse row and ~85% of a
dense one onto a single address.

Unlike the sampled threshold this keeps the generic two-pass structure
intact: the threshold is still exact and the collection still writes top_k
plus a small boundary bin, so there is no ranking of m * top_k candidates --
the term that sank sampling. Correctness needs only threshold bin < CLAMP;
the collection pass uses the same key, so folded elements land in a bin above
the threshold and are never taken.

What is NOT known: the 32x was measured with EVERY lane on one address. Here
one atomic instruction mixes a majority on one address with a minority
scattered, and nobody has measured that on this card. So this tests the
mechanism on the operator with FIXED clamps chosen per shape from the data,
before any per-row clamp estimate is designed.

The variants are the operator's _extract_bin_idx with only the STEP-0 line
changed; a gate at import diffs them against the operator's current source
and refuses to run if anything else differs.

Both arms are fresh copies of the GENERIC module at the override's geometry.
Production routes the many-row shapes through the prefix-sum copy, which
changes only the collection pass, so the histogram difference measured here
carries over; the absolute times on those shapes do not.

    tools/vendor_probe.sh tools/hygon_prefill_clamp.py hygon_prefill_clamp
"""

import difflib
import importlib.util
import pathlib
import sys
from importlib import import_module

import torch
import triton  # noqa: F401 -- the variants below are @triton.jit
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
_convert_to_uint32 = _generic._convert_to_uint32

SHAPES = [
    (64, 129280, 1024, 129280),
    (4, 16385, 512, 16648),
    (4, 8193, 512, 8456),
    (16383, 4095, 512, 4352),
    (12961, 4100, 512, 4360),
    (16380, 5115, 512, 5376),
    (4100, 1025, 512, 1288),
]
CLAMPS = (520, 540, 560, 600, 1100)
MARGIN = 4  # bins above the worst row's threshold
ROUNDS = 3


@triton.jit
def _extract_clamp_520(x, in_range, pattern, STEP: tl.constexpr):
    is_partial_match = in_range
    if STEP == 0:
        h = x.to(tl.float16)
        bits = h.to(tl.uint16, bitcast=True)
        sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
        sign_set = (bits & sign_mask) != 0
        inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
        mapped = tl.where(sign_set, bits, inv)
        bin_idx = tl.minimum(
            (mapped >> 5).to(tl.uint32), tl.full(mapped.shape, 520, tl.uint32)
        )
    else:
        bits = _convert_to_uint32(x)
        if STEP == 1:
            bin_idx = bits >> 21
        elif STEP == 2:
            bin_idx = (bits >> 10) & 0x7FF
            is_partial_match &= ((bits ^ pattern) >> 21) == 0
        elif STEP == 3:
            bin_idx = bits & 0x3FF
            is_partial_match &= ((bits ^ pattern) >> 10) == 0
    return bin_idx, is_partial_match


@triton.jit
def _extract_clamp_540(x, in_range, pattern, STEP: tl.constexpr):
    is_partial_match = in_range
    if STEP == 0:
        h = x.to(tl.float16)
        bits = h.to(tl.uint16, bitcast=True)
        sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
        sign_set = (bits & sign_mask) != 0
        inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
        mapped = tl.where(sign_set, bits, inv)
        bin_idx = tl.minimum(
            (mapped >> 5).to(tl.uint32), tl.full(mapped.shape, 540, tl.uint32)
        )
    else:
        bits = _convert_to_uint32(x)
        if STEP == 1:
            bin_idx = bits >> 21
        elif STEP == 2:
            bin_idx = (bits >> 10) & 0x7FF
            is_partial_match &= ((bits ^ pattern) >> 21) == 0
        elif STEP == 3:
            bin_idx = bits & 0x3FF
            is_partial_match &= ((bits ^ pattern) >> 10) == 0
    return bin_idx, is_partial_match


@triton.jit
def _extract_clamp_560(x, in_range, pattern, STEP: tl.constexpr):
    is_partial_match = in_range
    if STEP == 0:
        h = x.to(tl.float16)
        bits = h.to(tl.uint16, bitcast=True)
        sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
        sign_set = (bits & sign_mask) != 0
        inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
        mapped = tl.where(sign_set, bits, inv)
        bin_idx = tl.minimum(
            (mapped >> 5).to(tl.uint32), tl.full(mapped.shape, 560, tl.uint32)
        )
    else:
        bits = _convert_to_uint32(x)
        if STEP == 1:
            bin_idx = bits >> 21
        elif STEP == 2:
            bin_idx = (bits >> 10) & 0x7FF
            is_partial_match &= ((bits ^ pattern) >> 21) == 0
        elif STEP == 3:
            bin_idx = bits & 0x3FF
            is_partial_match &= ((bits ^ pattern) >> 10) == 0
    return bin_idx, is_partial_match


@triton.jit
def _extract_clamp_600(x, in_range, pattern, STEP: tl.constexpr):
    is_partial_match = in_range
    if STEP == 0:
        h = x.to(tl.float16)
        bits = h.to(tl.uint16, bitcast=True)
        sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
        sign_set = (bits & sign_mask) != 0
        inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
        mapped = tl.where(sign_set, bits, inv)
        bin_idx = tl.minimum(
            (mapped >> 5).to(tl.uint32), tl.full(mapped.shape, 600, tl.uint32)
        )
    else:
        bits = _convert_to_uint32(x)
        if STEP == 1:
            bin_idx = bits >> 21
        elif STEP == 2:
            bin_idx = (bits >> 10) & 0x7FF
            is_partial_match &= ((bits ^ pattern) >> 21) == 0
        elif STEP == 3:
            bin_idx = bits & 0x3FF
            is_partial_match &= ((bits ^ pattern) >> 10) == 0
    return bin_idx, is_partial_match


@triton.jit
def _extract_clamp_1100(x, in_range, pattern, STEP: tl.constexpr):
    is_partial_match = in_range
    if STEP == 0:
        h = x.to(tl.float16)
        bits = h.to(tl.uint16, bitcast=True)
        sign_mask = tl.full(bits.shape, 0x8000, tl.uint16)
        sign_set = (bits & sign_mask) != 0
        inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
        mapped = tl.where(sign_set, bits, inv)
        bin_idx = tl.minimum(
            (mapped >> 5).to(tl.uint32), tl.full(mapped.shape, 1100, tl.uint32)
        )
    else:
        bits = _convert_to_uint32(x)
        if STEP == 1:
            bin_idx = bits >> 21
        elif STEP == 2:
            bin_idx = (bits >> 10) & 0x7FF
            is_partial_match &= ((bits ^ pattern) >> 21) == 0
        elif STEP == 3:
            bin_idx = bits & 0x3FF
            is_partial_match &= ((bits ^ pattern) >> 10) == 0
    return bin_idx, is_partial_match


def _gate():
    """Each variant must differ from the operator's function only in its name
    and the STEP-0 bin line, which black spreads over three lines."""
    src = pathlib.Path(_generic.__file__).read_text()
    i = src.index("def _extract_bin_idx(")
    orig = src[i : src.index("\n\n\n", i)]
    me = pathlib.Path(__file__).read_text()

    def norm(x):
        return [ln.strip() for ln in x.splitlines() if ln.strip()]

    for c in CLAMPS:
        k = me.index(f"def _extract_clamp_{c}(")
        mine = me[k : me.index("\n\n\n", k)]
        d = [
            ln
            for ln in difflib.unified_diff(norm(orig), norm(mine), lineterm="")
            if ln[:1] in "+-" and ln[1:3] not in ("++", "--")
        ]
        want = [
            "-def _extract_bin_idx(x, in_range, pattern, STEP: tl.constexpr):",
            "-bin_idx = (mapped >> 5).to(tl.uint32)",
            f"+def _extract_clamp_{c}(x, in_range, pattern, STEP: tl.constexpr):",
            "+bin_idx = tl.minimum(",
            f"+(mapped >> 5).to(tl.uint32), tl.full(mapped.shape, {c}, tl.uint32)",
            "+)",
        ]
        assert sorted(d) == sorted(want), f"clamp {c} differs unexpectedly: {d}"


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
    spec = importlib.util.spec_from_file_location(name, _generic.__file__)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def keys(x):
    """The operator's STEP-0 key, on the host."""
    h = x.half().view(torch.int16).to(torch.int32) & 0xFFFF
    sign = (h & 0x8000) != 0
    inv = (~h) & 0x7FFF
    return torch.where(sign, h, inv) >> 5


def main():
    _gate()
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_prefill"
    )
    dev = "cuda"
    base = load_copy("flaggems_vllm.ops._topk_prefill_clamp_base")
    copies = {}
    print(
        "device us, unclamped against clamped, interleaved; both generic copies"
        " at the override's geometry\n"
    )
    print(
        f"  {'shape':>18} {'thr key':>8} {'clamp':>6} {'folded':>7} "
        f"{'plain':>9} {'clamped':>9} {'ratio':>7} {'spread':>13} {'ans':>5}"
    )
    for rows, vocab, top_k, stride0 in SHAPES:
        torch.manual_seed(42)
        buf = torch.randn((rows - 1) * stride0 + vocab, device=dev, dtype=torch.float32)
        logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
        starts = torch.zeros(rows, dtype=torch.int32, device=dev)
        ends = torch.full((rows,), vocab, dtype=torch.int32, device=dev)
        idx = torch.empty((rows, top_k), dtype=torch.int32, device=dev)
        vals = torch.topk(logits, top_k, dim=1).values
        want = vals.sort(dim=1).values
        thr = int(keys(vals[:, -1]).max())
        clamp = next((c for c in CLAMPS if c >= thr + MARGIN), None)
        if clamp is None:
            print(f"  {f'{rows}x{vocab}':>18} {thr:>8}  no clamp in {CLAMPS}")
            continue
        folded = float((keys(logits) > clamp).float().mean())
        if clamp not in copies:
            m = load_copy(f"flaggems_vllm.ops._topk_prefill_clamp_{clamp}")
            m._extract_bin_idx = globals()[f"_extract_clamp_{clamp}"]
            copies[clamp] = m
        mods = (base, copies[clamp])
        geo = ov._geometry(rows, vocab) if ov._GEOMETRY else None
        for m in mods:
            if geo is None:
                m.NUM_THREADS_PER_BLOCK = _generic.NUM_THREADS_PER_BLOCK
                m._num_warps = _generic._num_warps
            else:
                m.NUM_THREADS_PER_BLOCK = geo[0]
                m._num_warps = lambda b, w=geo[1]: w

        def runner(m):
            def go():
                m.top_k_per_row_prefill(
                    logits, starts, ends, idx, rows, stride0, 1, top_k
                )

            return go

        gos = [runner(m) for m in mods]
        ok = True
        for go in gos:
            idx.fill_(-9)
            go()
            torch.cuda.synchronize()
            got = logits.gather(1, idx.long().clamp(0, vocab - 1)).sort(dim=1).values
            ok = ok and torch.allclose(got, want) and bool((idx >= 0).all())
        pairs = []
        for _ in range(ROUNDS):
            pairs.append((device_us(gos[0]), device_us(gos[1])))
        rs = sorted(a / b for a, b in pairs)
        a = sorted(p[0] for p in pairs)[ROUNDS // 2]
        b = sorted(p[1] for p in pairs)[ROUNDS // 2]
        print(
            f"  {f'{rows}x{vocab}':>18} {thr:>8} {clamp:>6} {folded:>7.1%} "
            f"{a:>9.1f} {b:>9.1f} {rs[ROUNDS // 2]:>7.3f} "
            f"{rs[0]:>6.3f}-{rs[-1]:<6.3f} {'OK' if ok else 'WRONG':>5}",
            flush=True,
        )
    print(
        "\n  'folded' is the share of elements whose atomic goes to the single"
        "\n  overflow bin. A ratio well above 1 where that share is high means"
        "\n  mixed-address atomics do coalesce here, and a per-row clamp is"
        "\n  worth designing; near 1 means they do not, and this idea is done."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
