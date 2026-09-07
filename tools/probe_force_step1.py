"""Force STEP 1 of the radix pass to actually execute, then check correctness.

STEP 0 buckets fp16 bits into 2048 bins and stops there unless the threshold
bin holds more than NUM_FINAL_ITEMS=2048 candidates:

    return final_bin_size > NUM_FINAL_ITEMS, ...

So randn/129280/top_k=1024 -- what the suite runs -- puts ~200 in that bin and
STEP 1-3 never run. Their `bits >> 21` on a genuine tl.uint32 is therefore
never exercised, which is why the suite passes despite the arithmetic-shift
defect (FlagTree #1121).

This probe concentrates the values into ONE fp16 bucket so the bin overflows
2048 and STEP 1 must run, and reports the measured bucket population so
"did STEP 1 run" is observed rather than assumed. Sign is varied separately:
_convert_to_uint32 leaves the top bit set for NEGATIVE inputs only, so if the
defect reaches STEP 1 the negative case fails and the positive one does not.

    python tools/probe_force_step1.py
"""

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass

import flaggems_vllm

DEV = "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda"
SYNC = torch.npu.synchronize if DEV == "npu" else torch.cuda.synchronize
VOCAB, TOPK, ROWS = 129280, 1024, 2
NUM_FINAL_ITEMS = 2048


def step0_bucket(x):
    """Reproduce STEP 0's bin index on the CPU, exactly as the kernel does."""
    bits = x.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    sign_set = (bits & 0x8000) != 0
    inv = (~bits) & 0x7FFF
    mapped = torch.where(sign_set, bits, inv)
    return mapped >> 5


def threshold_bin_size(logits, top_k):
    """How many values share the bucket of the k-th largest -- the STEP 1 gate."""
    worst = 0
    for row in logits.cpu():
        kth = torch.topk(row, top_k).values[-1]
        buckets = step0_bucket(row)
        worst = max(worst, int((buckets == step0_bucket(kth.reshape(1))[0]).sum()))
    return worst


def case(name, logits):
    logits = logits.contiguous()
    binsz = threshold_bin_size(logits, TOPK)
    runs_step1 = binsz > NUM_FINAL_ITEMS

    row_starts = torch.zeros(ROWS, dtype=torch.int32, device=DEV)
    row_ends = torch.full((ROWS,), VOCAB, dtype=torch.int32, device=DEV)
    out = torch.empty((ROWS, TOPK), dtype=torch.int32, device=DEV)
    flaggems_vllm.top_k_per_row_prefill(
        logits, row_starts, row_ends, out, ROWS,
        logits.stride(0), logits.stride(1), TOPK,
    )
    SYNC()

    ref = torch.topk(logits, TOPK, dim=-1).indices
    got_v = torch.gather(logits, 1, out.long()).sort(dim=-1, descending=True).values
    ref_v = torch.gather(logits, 1, ref).sort(dim=-1, descending=True).values
    ok = torch.equal(got_v, ref_v)

    print(f"{name:32} bin {binsz:>7} | STEP1 {'YES' if runs_step1 else 'no ':>3}"
          f" | {'PASS' if ok else 'FAIL'}"
          f"{'' if ok else f'  max |diff| {(got_v - ref_v).abs().max().item():.3e}'}")
    return runs_step1, ok


def main():
    torch.manual_seed(42)
    print(f"vocab {VOCAB}, top_k {TOPK}, {ROWS} rows, device {DEV}")
    print(f"STEP 1 runs only when the threshold bin exceeds {NUM_FINAL_ITEMS}\n")

    def tight(centre, spread):
        u = torch.rand(ROWS, VOCAB, dtype=torch.float32)
        return (centre + (u - 0.5) * spread).to(DEV)

    results = {}
    # Baseline: the suite's own distribution. STEP 1 must NOT run here.
    results["randn"] = case(
        "randn (suite baseline)",
        torch.randn(ROWS, VOCAB, dtype=torch.float32).to(DEV))

    # Concentrated into one fp16 bucket -> the bin overflows and STEP 1 runs.
    # A single fp16 bucket near 1.5 is about 1.5/32 = 0.047 wide; stay inside it.
    results["pos"] = case("tight POSITIVE (+1.51 +/- 0.02)", tight(1.51, 0.04))
    results["neg"] = case("tight NEGATIVE (-1.51 +/- 0.02)", tight(-1.51, 0.04))

    print()
    forced = results["pos"][0] and results["neg"][0]
    if not forced:
        print("INCONCLUSIVE: the tight cases did not overflow the threshold bin,")
        print("so STEP 1 still never ran. Narrow the spread and retry.")
    elif results["pos"][1] and not results["neg"][1]:
        print("CONFIRMED: STEP 1 ran in both, correct for positive values and")
        print("wrong for negative ones -- the uint32 arithmetic shift is live in")
        print("our own STEP 1-3 and needs the same widen-and-mask workaround.")
    elif results["pos"][1] and results["neg"][1]:
        print("STEP 1 ran and both passed. On a kernel that still has the raw")
        print("`bits >> 21` this would mean the defect does not reach STEP 1 --")
        print("check that before concluding. With the widen-and-mask fix in")
        print("_extract_bin_idx already applied, this is the expected result.")
    else:
        print("Both concentrated cases failed -- that is not the sign-dependent")
        print("signature of the shift defect. Investigate STEP 1 generally.")


if __name__ == "__main__":
    main()
