"""How much would splitting a decode row across programs actually buy?

Measured on C550: the non-TLE decode path launches `[(num_rows,)]` -- one
program per row, no intra-row parallelism at any vocab size. At num_rows=1 that
is 1 program on a 104-SM card, and the single kernel takes 0.335 ms of device
time against vLLM's 0.088 ms in a split-and-merge pair.

Before writing a merge kernel, measure the ceiling. A row split into S chunks
can be fed to the EXISTING kernel as S rows, because the global top-k of a row
is contained in the union of its chunks' top-k: if x is in the global top-512,
at most 511 elements of its own chunk exceed it, so x is in that chunk's top-512.
So stage 1 needs no new kernel at all -- only the merge does.

Reports stage-1 device time, host-merge time, and the total against both the
unsplit path and vLLM, so the decision rests on a measurement rather than on
the arithmetic that says the headroom is large.

    python tools/decode_split_ceiling.py
"""

import sys

import torch

import flaggems_vllm

DEV = flaggems_vllm.device
VOCAB, TOPK = 262144, 512
# 262144 is 2**18, so only powers of two divide it evenly. 128 chunks puts
# 128 programs on the 104 SMs at num_rows=1 -- just past one full wave.
SPLITS = (1, 8, 16, 32, 64, 128, 256)
ROWS = (1, 8)

try:
    import vllm._custom_ops  # noqa: F401
    HAS_VLLM = hasattr(torch.ops._C, "top_k_per_row_decode")
except Exception:  # noqa: BLE001
    HAS_VLLM = False


def timed(fn, iters=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def decode(logits, seq_lens, out, rows):
    flaggems_vllm.top_k_per_row_decode(
        logits, 1, seq_lens, out, rows,
        logits.stride(0), logits.stride(1), TOPK)


def main():
    print(f"decode split ceiling | vocab={VOCAB} top_k={TOPK} | device {DEV}")
    print("stage 1 reuses the existing kernel on a reshaped view; only the")
    print("merge would be new work.\n")

    for rows in ROWS:
        logits = torch.randn(rows, VOCAB, device=DEV, dtype=torch.float32)
        print("=" * 76)
        print(f"=== num_rows = {rows}   (grid today = {rows} programs, card has 104 SMs)")
        print("=" * 76)

        sl = torch.full((rows,), VOCAB, dtype=torch.int32, device=DEV)
        out = torch.empty((rows, TOPK), dtype=torch.int32, device=DEV)
        base = timed(lambda: decode(logits, sl, out, rows))
        ref = torch.topk(logits, TOPK, dim=-1).values.sort(dim=-1, descending=True).values
        print(f"  today, unsplit                       {base:8.4f} ms")

        if HAS_VLLM:
            o2 = torch.empty((rows, TOPK), dtype=torch.int32, device=DEV)
            v = timed(lambda: torch.ops._C.top_k_per_row_decode(
                logits, 1, sl, o2, rows, logits.stride(0), logits.stride(1), TOPK))
            print(f"  vLLM mcoplib                         {v:8.4f} ms")

        print(f"\n  {'split':>6} {'chunk':>8} {'stage1':>9} {'gather':>8} {'t.topk':>8} "
              f"{'ourkern':>9} {'total':>8} {'vs now':>7}  correct")
        for s in SPLITS:
            if VOCAB % s or (VOCAB // s) < TOPK:
                continue
            chunk = VOCAB // s
            view = logits.reshape(rows * s, chunk).contiguous()
            n2 = rows * s
            sl2 = torch.full((n2,), chunk, dtype=torch.int32, device=DEV)
            out2 = torch.empty((n2, TOPK), dtype=torch.int32, device=DEV)

            stage1 = timed(lambda: decode(view, sl2, out2, n2))

            def gather():
                return torch.gather(view, 1, out2.long())          # [rows*s, TOPK]
            g = timed(gather)
            cand = gather().reshape(rows, s * TOPK).contiguous()

            def select():
                return torch.topk(cand, TOPK, dim=-1)
            sel = timed(select)

            # What the SAME kernel costs on the candidate array -- this is the
            # merge a real second pass would run, rather than torch.topk.
            if cand.shape[1] >= TOPK:
                sl3 = torch.full((rows,), cand.shape[1], dtype=torch.int32, device=DEV)
                out3 = torch.empty((rows, TOPK), dtype=torch.int32, device=DEV)
                k2 = timed(lambda: decode(cand, sl3, out3, rows))
            else:
                k2 = float("nan")

            got = select().values.sort(dim=-1, descending=True).values
            ok = torch.allclose(got, ref, atol=1e-6, rtol=1e-6)
            tot = stage1 + g + k2
            print(f"  {s:>6} {chunk:>8} {stage1:>9.4f} {g:>8.4f} {sel:>8.4f} "
                  f"{k2:>9.4f} {tot:>8.4f} {base / tot:>7.2f}x  "
                  f"{'yes' if ok else 'NO'}")
        print()

    print("total = stage1 + gather + ourkern, i.e. a two-pass merge built from the")
    print("kernel we already have. t.topk is torch's selector on the same candidates,")
    print("shown only for contrast. A fused merge would do the gather inside the")
    print("kernel, so `total` is still an upper bound -- but a much tighter one.")


if __name__ == "__main__":
    sys.exit(main())
