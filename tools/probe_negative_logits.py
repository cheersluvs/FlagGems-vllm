"""Does the uint32 arithmetic-shift defect reach top_k_per_row_prefill?

STEP 1-3 of the radix pass bucket by `bits >> 21` on a genuine tl.uint32.
`_convert_to_uint32` leaves the top bit SET for negative inputs only, so an
arithmetic shift corrupts the bucket of negatives and leaves positives intact.

The suite uses randn with top_k=1024 of 129280 -- a threshold near +2.5 sigma,
so every selected value is positive and the corrupted half is never consulted.
This probe moves the threshold into the negatives and checks whether the answer
survives.

    python tools/probe_negative_logits.py
"""

import torch

try:  # torch.npu only exists once torch_npu is imported
    import torch_npu  # noqa: F401
except ImportError:
    pass

import flaggems_vllm

DEV = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
SYNC = torch.npu.synchronize if DEV == "npu" else torch.cuda.synchronize
VOCAB, TOPK = 129280, 1024


def reference(logits, top_k):
    return torch.topk(logits, top_k, dim=-1).indices.to(torch.int32)


def values_match(logits, got, ref, top_k):
    """Compare selected VALUES, not indices -- ties make indices ambiguous."""
    gv = torch.gather(logits, 1, got.long()).sort(dim=-1, descending=True).values
    rv = torch.gather(logits, 1, ref.long()).sort(dim=-1, descending=True).values
    return torch.allclose(gv, rv)


def case(name, logits):
    num_rows = logits.shape[0]
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device=DEV)
    row_ends = torch.full((num_rows,), VOCAB, dtype=torch.int32, device=DEV)
    out = torch.empty((num_rows, TOPK), dtype=torch.int32, device=DEV)

    flaggems_vllm.top_k_per_row_prefill(
        logits, row_starts, row_ends, out, num_rows,
        logits.stride(0), logits.stride(1), TOPK,
    )
    SYNC()

    ref = reference(logits, TOPK)
    ok = values_match(logits, out, ref, TOPK)

    kth = torch.topk(logits, TOPK, dim=-1).values[:, -1]
    frac_neg = (logits < 0).float().mean().item()
    print(f"{name:34} threshold {kth.min().item():+8.4f} .. {kth.max().item():+8.4f}"
          f" | {frac_neg*100:5.1f}% negative | {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    torch.manual_seed(42)
    n = 4

    print(f"vocab {VOCAB}, top_k {TOPK}, {n} rows, device {DEV}")
    print("the threshold column is the k-th largest value -- the sign that matters\n")

    results = []
    # Control: exactly what the suite runs.  Threshold is positive, so the
    # corrupted negative half is never consulted.
    results.append(case("randn (what the suite runs)",
                        torch.randn(n, VOCAB, device=DEV, dtype=torch.float32)))

    # Shift the whole distribution down so the k-th largest is negative.
    # Every selected value now comes from the half with the top bit set.
    results.append(case("randn - 3.0 (threshold negative)",
                        torch.randn(n, VOCAB, device=DEV, dtype=torch.float32) - 3.0))

    # Every value negative, nothing else changed.
    results.append(case("all negative",
                        -torch.rand(n, VOCAB, device=DEV, dtype=torch.float32) * 10 - 1))

    print()
    if results[0] and not all(results[1:]):
        print("CONFIRMED: correct when the top-k is positive, wrong once the")
        print("threshold crosses into the negatives -- the uint32 shift defect")
        print("reaches this operator and the suite simply never looks there.")
    elif all(results):
        print("All pass -- the defect does NOT reach this operator; something")
        print("else protects STEP 1-3.  Do not claim the shift is logical here.")
    else:
        print("Unexpected pattern -- read the rows above before concluding.")


if __name__ == "__main__":
    main()
