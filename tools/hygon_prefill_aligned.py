"""Is prefill's aligned fast path reachable, and is it worth reaching?

The kernel takes a fast path when

    row_start == 0 and row_end == vocab_size and stride1 == 1
    and vocab_size % BLOCK_SIZE == 0

otherwise every row, in every refinement step, walks a rem_tiles loop and a
rem_elems remainder. Of the benchmark's seven shapes only ONE can ever satisfy
the modulo: 129280 = 256 x 505, so BLOCK 128 or 256 aligns it and 512 does
not. The other six have odd vocabularies (8193, 4095, 16385, 5115, 1025) or
4100, which no power of two divides.

And the geometry rule hands (64,129280) BLOCK 512 -- it returns None below four
rows per SM, so the shape falls back to the generic default. That shape is
prefill's worst ratio. So the question is whether the rule is holding it out
of a fast path that exists only for it.

Two measurements, both interleaved, because this shape is bimodal: its device
time has come back 228, 232, 238, 311 and 362 us across probes, in two
clusters, so a single before/after pair proves nothing.

  the sweep     BLOCK 128/256/512/1024 at that block's natural warp count,
                with whether each one aligns, to see if an aligned block is
                simply faster overall
  the isolation BLOCK fixed at 256, row_end = vocab against row_end = vocab-1.
                One element less, same geometry, same work -- but the second
                fails the alignment test. That difference IS the tail path's
                cost, with nothing else moving.

    tools/vendor_probe.sh tools/hygon_prefill_aligned.py hygon_prefill_aligned
"""

import sys
from importlib import import_module

import torch
from torch.profiler import ProfilerActivity, profile

_generic = import_module("flaggems_vllm.ops.top_k_per_row_prefill")

ROWS, VOCAB, TOPK, STRIDE0 = 64, 129280, 1024, 129280
BLOCKS = (128, 256, 512, 1024)
ROUNDS = 5


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


def interleave(a, b, rounds=ROUNDS):
    """Alternate two closures; return the ratios a/b, in order."""
    out = []
    for _ in range(rounds):
        ta = device_us(a)
        tb = device_us(b)
        out.append((ta, tb))
    return out


def main():
    ov = import_module(
        "flaggems_vllm.runtime.backend._hygon.fused.top_k_per_row_decode"
    )
    gen = _generic
    dev = "cuda"
    torch.manual_seed(42)
    buf = torch.randn((ROWS - 1) * STRIDE0 + VOCAB, device=dev, dtype=torch.float32)
    logits = torch.as_strided(buf, (ROWS, VOCAB), (STRIDE0, 1))
    starts = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    ends_full = torch.full((ROWS,), VOCAB, dtype=torch.int32, device=dev)
    ends_short = torch.full((ROWS,), VOCAB - 1, dtype=torch.int32, device=dev)
    idx = torch.empty((ROWS, TOPK), dtype=torch.int32, device=dev)

    def scratch(n):
        return (
            torch.empty((n, gen.NUM_BINS), dtype=torch.int32, device=dev),
            torch.empty((n, gen.NUM_FILNAL_ITEMS), dtype=torch.float32, device=dev),
            torch.empty((n,), dtype=torch.int32, device=dev),
            torch.empty((n,), dtype=torch.int32, device=dev),
            torch.empty((n,), dtype=torch.int32, device=dev),
            torch.empty((n,), dtype=torch.int32, device=dev),
        )

    sc = scratch(ROWS)

    def make(block, warps):
        return ov._Launch(
            gen.non_tle_top_k_per_row_prefill,
            (ROWS,),
            {"TOPK": TOPK, "BLOCK_SIZE": block, "ROW_OFFSET": 0},
            warps,
        )

    def call(lj, ends):
        def go():
            lj(logits, idx, starts, ends, STRIDE0, 1, VOCAB, *sc)

        return go

    def correct(ends):
        n = int(ends[0])
        want = torch.topk(logits[:, :n], TOPK, dim=1).values.sort(dim=1).values
        got = logits.gather(1, idx.long().clamp(0, n - 1)).sort(dim=1).values
        return torch.allclose(got, want) and bool((idx >= 0).all())

    print(f"{ROWS} x {VOCAB}, top_k {TOPK}\n")
    print(f"  {'BLOCK':>6} {'warps':>6} {'aligns':>7} {'device us':>10} {'ans':>5}")
    for block in BLOCKS:
        warps = gen._num_warps(block)
        lj = make(block, warps)
        go = call(lj, ends_full)
        idx.fill_(-9)
        try:
            go()
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 - keep sweeping
            print(f"  {block:>6} {warps:>6}  FAILED {exc!r:.50}", flush=True)
            continue
        ok = correct(ends_full)
        t = device_us(go)
        print(
            f"  {block:>6} {warps:>6} {'yes' if VOCAB % block == 0 else 'no':>7} "
            f"{t:>10.1f} {'OK' if ok else 'WRONG':>5}",
            flush=True,
        )

    print(
        "\n  isolation at BLOCK 256: row_end = vocab (aligns) against"
        " vocab-1 (does not),\n  one element apart, interleaved\n"
    )
    warps = gen._num_warps(256)
    lj = make(256, warps)
    idx.fill_(-9)
    call(lj, ends_full)()
    torch.cuda.synchronize()
    ok_a = correct(ends_full)
    idx.fill_(-9)
    call(lj, ends_short)()
    torch.cuda.synchronize()
    ok_b = correct(ends_short)
    pairs = interleave(call(lj, ends_full), call(lj, ends_short))
    for ta, tb in pairs:
        print(
            f"      aligned {ta:>8.1f}   tail {tb:>8.1f}   tail/aligned {tb / ta:>6.3f}"
        )
    rs = sorted(tb / ta for ta, tb in pairs)
    print(
        f"      median {rs[len(rs) // 2]:.3f}, spread {rs[0]:.3f} to {rs[-1]:.3f}; "
        f"answers {'OK' if ok_a and ok_b else 'WRONG'}"
    )
    print(
        "\n  A tail/aligned well above 1.0 means the fast path is worth"
        " reaching, and\n  only the geometry rule (which gives this shape"
        " BLOCK 512) stands in the way."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
