"""CPU-only check of the MetaX decode override's tensor plumbing.

The override is mostly reshapes, gathers and index arithmetic wrapped around
two calls to the real kernel. Those wrappers can be wrong in ways that have
nothing to do with the GPU -- a broadcast that does not broadcast, an index
space that does not line up, padding that does not propagate -- and finding
them through the full suite costs a GPU round trip per mistake.

So run the same code on the CPU with the kernel replaced by torch.topk. It
exercises every line of the override except the two launches, in seconds, and
it fails loudly on exactly the class of bug that shape errors belong to.

    python tools/metax_decode_plumbing.py
"""

import sys

from importlib import import_module

import torch

# NOT `import ...top_k_per_row_decode as ov`: that resolves by attribute lookup
# on the parent package, and _metax/fused/__init__.py exports a FUNCTION under
# that same name, which shadows the module.
ov = import_module("flaggems_vllm.runtime.backend._metax.fused.top_k_per_row_decode")


def fake_kernel(logits, next_n, seq_lens, indices, num_rows, s0, s1, top_k):
    """Stand-in with the real kernel's contract, including its -1 padding."""
    assert logits.shape[0] == num_rows, (logits.shape, num_rows)
    assert s1 == 1 and s0 == logits.shape[1], (s0, s1, logits.shape)
    width = logits.shape[1]
    for r in range(num_rows):
        n = int(seq_lens[r])
        indices[r].fill_(-1)
        if n <= 0:
            continue
        k = min(top_k, n)
        indices[r, :k] = torch.topk(logits[r, :n], k).indices.to(torch.int32)
    assert width >= 0
    return indices


def reference(logits, seq_lens, top_k):
    rows = logits.shape[0]
    out = torch.full((rows, top_k), -1, dtype=torch.int32)
    for r in range(rows):
        n = int(seq_lens[r])
        k = min(top_k, n)
        if k > 0:
            out[r, :k] = torch.topk(logits[r, :n], k).indices.to(torch.int32)
    return out


def values_of(logits, idx):
    """Compare by VALUE, not index -- ties make the index ambiguous."""
    rows = logits.shape[0]
    got = []
    for r in range(rows):
        v = [logits[r, int(i)].item() for i in idx[r] if int(i) >= 0]
        got.append(sorted(v, reverse=True))
    return got


def main():
    torch.manual_seed(0)
    ov._sm_count.cache_clear()
    ov._sm_count = lambda: 104          # pin the geometry the card reports
    real, ov._generic.top_k_per_row_decode = ov._generic.top_k_per_row_decode, fake_kernel

    cases = [
        # rows, vocab, top_k, seq_lens             what it covers
        (1, 65536, 512, None),                    # the split path, one row
        (8, 65536, 512, None),                    # several rows
        (4, 65536, 512, [65536, 40000, 8, 0]),    # short rows, empty chunks, empty row
        (2, 65536, 512, [512, 511]),              # exactly and just under top_k
        (3, 4096, 512, None),                     # below 2*MIN_CHUNK -> fallback
        (1, 65536, 16384, None),                  # top_k > MIN_CHUNK -> fallback
    ]

    bad = 0
    for rows, vocab, top_k, sl in cases:
        logits = torch.randn(rows, vocab, dtype=torch.float32)
        seq_lens = (torch.full((rows,), vocab, dtype=torch.int32)
                    if sl is None else torch.tensor(sl, dtype=torch.int32))
        out = torch.empty((rows, top_k), dtype=torch.int32)
        split = ov._split_factor(rows, vocab)
        try:
            ov.top_k_per_row_decode(logits, 1, seq_lens, out,
                                    rows, vocab, 1, top_k)
            ok = values_of(logits, out) == values_of(logits, reference(logits, seq_lens, top_k))
            verdict = "ok" if ok else "WRONG VALUES"
        except Exception as exc:  # noqa: BLE001 - the failure is the result
            verdict = f"RAISED {type(exc).__name__}: {exc}"
            ok = False
        bad += not ok
        print(f"  rows={rows:<3} vocab={vocab:<7} top_k={top_k:<6} "
              f"split={split:<4} {verdict}")

    ov._generic.top_k_per_row_decode = real
    print()
    print("all plumbing cases pass" if not bad else f"{bad} case(s) FAILED")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
