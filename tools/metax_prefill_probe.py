"""Where does prefill lose on the C550, and is splitting the answer here?

Measured ratios against vLLM put three different problems in one table:

  (4100,1025)   0.406   1025 elements per row against a 2048-bin histogram --
                        30 GB/s where every other full-grid shape reaches 92+
  (12961,4100)  0.528   we run at 91.8 GB/s, the same as the 4095 shape next
                        to it; vLLM runs at 173 there and 114 here, so this
                        one is vLLM being fast, not us being slow
  (64,129280)   0.546   production shape, 64 programs on 104 SMs, and the only
                        shape whose rows are contiguous

So: does splitting help the production shape, the way it did for decode? The
memory from Moore Threads says row-splitting prefill was refuted at every grid
size, the ceiling being the four-step serial chain rather than occupancy -- but
that was a different card on the TLE path, so measure rather than inherit.

Stage one only; the merge would cost extra, so a split that does not win here
cannot win at all.

    python tools/metax_prefill_probe.py
"""

import sys

import torch

import flaggems_vllm

DEV = flaggems_vllm.device

try:
    import vllm._custom_ops  # noqa: F401
    HAS_VLLM = hasattr(torch.ops._C, "top_k_per_row_prefill")
except Exception:  # noqa: BLE001
    HAS_VLLM = False


def timed(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def make(rows, vocab, stride0, top_k):
    buf = torch.randn((rows - 1) * stride0 + vocab, device=DEV, dtype=torch.float32)
    logits = torch.as_strided(buf, (rows, vocab), (stride0, 1))
    starts = torch.zeros(rows, dtype=torch.int32, device=DEV)
    ends = torch.full((rows,), vocab, dtype=torch.int32, device=DEV)
    out = torch.empty((rows, top_k), dtype=torch.int32, device=DEV)
    return buf, logits, starts, ends, out


def call(logits, starts, ends, out, rows, top_k):
    return lambda: flaggems_vllm.top_k_per_row_prefill(
        logits, starts, ends, out, rows, logits.stride(0), logits.stride(1), top_k)


def gbps(rows, vocab, ms):
    return rows * vocab * 4 / 1e9 / (ms / 1000)


def main():
    print(f"device {DEV} | 104 SMs\n")

    print("=" * 78)
    print("=== every benchmark shape, as achieved bandwidth")
    print("=" * 78)
    print(f"  {'shape':>16} {'k':>5} {'rows/SM':>8} {'ms':>9} {'GB/s':>8} "
          f"{'vLLM ms':>9} {'vLLM GB/s':>10} {'ratio':>7}")
    for rows, vocab, top_k, stride0 in (
        (64, 129280, 1024, 129280), (4, 8193, 512, 8456),
        (16383, 4095, 512, 4352), (4, 16385, 512, 16648),
        (12961, 4100, 512, 4360), (16380, 5115, 512, 5376),
        (4100, 1025, 512, 1288),
    ):
        _b, lg, st, en, out = make(rows, vocab, stride0, top_k)
        t = timed(call(lg, st, en, out, rows, top_k))
        v = vv = float("nan")
        if HAS_VLLM:
            o2 = torch.empty_like(out)
            v = timed(lambda: torch.ops._C.top_k_per_row_prefill(
                lg, st, en, o2, rows, lg.stride(0), lg.stride(1), top_k))
            vv = gbps(rows, vocab, v)
        print(f"  {f'({rows},{vocab})':>16} {top_k:>5} {rows / 104:>8.2f} {t:>9.4f} "
              f"{gbps(rows, vocab, t):>8.1f} {v:>9.4f} {vv:>10.1f} {v / t:>7.3f}")

    print()
    print("=" * 78)
    print("=== does splitting the production shape help? (stage one only)")
    print("=" * 78)
    rows, vocab, top_k = 64, 129280, 1024
    _b, lg, st, en, out = make(rows, vocab, vocab, top_k)
    base = timed(call(lg, st, en, out, rows, top_k))
    print(f"  unsplit: {base:.4f} ms, {rows} programs, {gbps(rows, vocab, base):.1f} GB/s")
    print(f"  {'split':>6} {'chunk':>8} {'prog':>6} {'stage1 ms':>10} {'vs unsplit':>11}")
    for s in (2, 4, 8, 16, 32):
        if vocab % s:
            continue
        chunk = vocab // s
        view = lg.reshape(rows * s, chunk)
        st2 = torch.zeros(rows * s, dtype=torch.int32, device=DEV)
        en2 = torch.full((rows * s,), chunk, dtype=torch.int32, device=DEV)
        o2 = torch.empty((rows * s, top_k), dtype=torch.int32, device=DEV)
        t = timed(call(view, st2, en2, o2, rows * s, top_k))
        print(f"  {s:>6} {chunk:>8} {rows * s:>6} {t:>10.4f} {base / t:>10.2f}x")
    print("\n  A split that does not beat 1.00x here cannot pay for a merge on top.")


if __name__ == "__main__":
    sys.exit(main())
