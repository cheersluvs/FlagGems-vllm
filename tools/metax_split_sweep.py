"""What split factor does each decode shape actually want?

The automatic rule was fitted when the bookkeeping between the two passes cost
more than the passes did, and it has never been re-measured against the fused
kernels. It currently refuses to split at all above 13 rows -- which is where
every remaining loss to vLLM lives.

So measure the surface instead of arguing about it: for every benchmark shape,
time the override at every legal split, next to vLLM's own kernel.

    python tools/metax_split_sweep.py
    python tools/metax_split_sweep.py 16 32      # just these row counts
"""

import os
import sys

import torch

import flaggems_vllm

DEV = flaggems_vllm.device
VOCAB, TOPK = 262144, 512
MIN_CHUNK = 8192
ROWS = (1, 4, 8, 16, 24, 32, 40, 48, 56, 496, 512)

try:
    import vllm._custom_ops  # noqa: F401
    HAS_VLLM = hasattr(torch.ops._C, "top_k_per_row_decode")
except Exception:  # noqa: BLE001
    HAS_VLLM = False


def timed(fn, iters=30, warmup=10):
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


def legal_splits(vocab, top_k):
    s, out = 1, []
    while s <= vocab:
        # Below top_k a chunk cannot even produce its own top-k; below
        # MIN_CHUNK the 2048-bin histogram costs more than the data. Keep one
        # step past MIN_CHUNK so the sweep can confirm that bound rather than
        # inherit it.
        if vocab % s == 0 and vocab // s >= max(top_k, MIN_CHUNK // 2):
            out.append(s)
        s *= 2
    return out


def main():
    wanted = [int(a) for a in sys.argv[1:]] or list(ROWS)
    print(f"device {DEV} | vocab {VOCAB} top_k {TOPK} | 104 SMs")
    print("ratio is vLLM / gems, so >= 1.00 means we are ahead. Target 0.95.\n")

    for rows in wanted:
        logits = torch.randn(rows, VOCAB, device=DEV, dtype=torch.float32)
        sl = torch.full((rows,), VOCAB, dtype=torch.int32, device=DEV)
        out = torch.empty((rows, TOPK), dtype=torch.int32, device=DEV)
        args = (logits, 1, sl, out, rows, logits.stride(0), logits.stride(1), TOPK)

        ref = torch.topk(logits, TOPK, dim=-1).values.sort(-1, descending=True).values
        base = None
        if HAS_VLLM:
            o2 = torch.empty((rows, TOPK), dtype=torch.int32, device=DEV)
            base = timed(lambda: torch.ops._C.top_k_per_row_decode(
                logits, 1, sl, o2, rows, logits.stride(0), logits.stride(1), TOPK))

        print("=" * 74)
        print(f"=== {rows} rows" + (f"   vLLM {base:.4f} ms" if base else ""))
        print(f"  {'split':>6} {'chunk':>8} {'prog':>6} {'ms':>9} {'ratio':>7}  ok")
        best = None
        for s in legal_splits(VOCAB, TOPK):
            os.environ["FLAGGEMS_METAX_TOPK_SPLIT"] = str(s)
            try:
                t = timed(lambda: flaggems_vllm.top_k_per_row_decode(*args))
                got = torch.gather(logits, 1, out.long()).sort(-1, descending=True).values
                ok = torch.allclose(got, ref, atol=1e-6, rtol=1e-6)
            except Exception as exc:  # noqa: BLE001 - a failing split is a result
                print(f"  {s:>6} {VOCAB // s:>8} {rows * s:>6} "
                      f"{'FAILED':>9}  {type(exc).__name__}")
                continue
            r = (base / t) if base else float("nan")
            if ok and (best is None or t < best[1]):
                best = (s, t, r)
            print(f"  {s:>6} {VOCAB // s:>8} {rows * s:>6} {t:>9.4f} {r:>7.3f}  "
                  f"{'yes' if ok else 'WRONG'}")
        os.environ.pop("FLAGGEMS_METAX_TOPK_SPLIT", None)
        auto = timed(lambda: flaggems_vllm.top_k_per_row_decode(*args))
        ar = (base / auto) if base else float("nan")
        print(f"  {'auto':>6} {'':>8} {'':>6} {auto:>9.4f} {ar:>7.3f}")
        if best:
            print(f"  -> best split {best[0]}: {best[1]:.4f} ms, ratio {best[2]:.3f}"
                  f"{'  (auto already there)' if abs(best[1] - auto) / auto < 0.02 else '  AUTO IS OFF'}")
        print()


if __name__ == "__main__":
    main()
