"""Where does prefill's per-element cost go, and is tl.histogram cheaper here?

After the slot-allocation override, prefill's remaining cost on this card is
dominated by a term linear in vocab, ~1.3-1.9 ns per element on every shape --
~172 of ~186 us per program on the production (64, 129280). The candidate is
the histogram: `_distribute_to_bins` does one GLOBAL atomic per element into
2048 counters, and 4096 such atomics measured 6.66 us here (~1.6 ns each).

tl.histogram is the obvious replacement and is unmeasured on BW1000. It is not
safe to guess: the same code measured 5x SLOWER on MetaX C550, 3x faster on
Ascend and 1.42x faster on Moore Threads.

Each mode below runs the operator's own loop shape -- [512, VEC=4] tiles, the
operator's own STEP-0 bin extraction (imported, not re-derived) -- over 65536
elements per program, so per-element costs dominate the launch floor:

    load              read the tile                            (floor)
    extract           + bin index                              (what every pass pays)
    atomic            + clear, masked global atomic scatter    (today, step 0)
    histogram         + tl.histogram per tile, accumulated in registers
    atomic_masked     atomic scatter with a half-density mask  (steps 1-3 shape)
    histogram_masked  tl.histogram with the same mask
    pass2_sparse      + a second pass: per-element slot atomic at ~1/64 density,
                      masked store -- _process_bins on a sparse row

`atomic` and `histogram` (and the two masked modes) must produce identical
counts; that is checked.

    tools/vendor_probe.sh tools/hygon_histogram_cost.py hygon_histogram_cost
"""

import sys
from importlib import import_module

import torch
import triton
import triton.language as tl

import flaggems_vllm  # noqa: F401

_gen = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
_extract_bin_idx = _gen._extract_bin_idx

SMS = int(getattr(torch.cuda.get_device_properties(0), "multi_processor_count", 80))
ROWS = 10 * SMS  # 10 waves: 65536 elements per program is heavy enough
VOCAB = 65536
BLOCK = 512
VEC = 4
NB = 2048
K = 1024
WARPS = 8


@triton.jit
def k_hist(
    logits_ptr,
    hist_ptr,
    out_ptr,
    chk_ptr,
    MODE: tl.constexpr,
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
    VEC: tl.constexpr,
    NB: tl.constexpr,
    K: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    vec = tl.arange(0, VEC)
    bins = tl.arange(0, NB)
    ones = tl.full([BLOCK, VEC], 1, tl.int32)
    base_ptr = logits_ptr + row * VOCAB
    hrow = hist_ptr + row * NB
    if (MODE == 2) | (MODE == 4):
        tl.store(hrow + bins, tl.zeros([NB], tl.int32))
        tl.debug_barrier()
    acc = tl.zeros([NB], tl.int32)
    chk = tl.zeros([], tl.float32)
    if MODE == 6:
        tl.store(chk_ptr + row, 0)
        tl.debug_barrier()
    n_tiles = VOCAB // (BLOCK * VEC)
    for t in tl.range(0, n_tiles):
        offs = (t * BLOCK * VEC + lane * VEC)[:, None] + vec[None, :]
        x = tl.load(base_ptr + offs)
        if MODE == 0:
            chk += tl.sum(x)
        else:
            bin_idx, _ = _extract_bin_idx(x, True, 0, STEP=0)
            bi = bin_idx.to(tl.int32)
            if MODE == 1:
                chk += tl.sum(bi).to(tl.float32)
            elif MODE == 2:
                tl.atomic_add(hrow + bi, ones, sem="relaxed", scope="cta")
            elif MODE == 3:
                acc += tl.histogram(tl.reshape(bi, (BLOCK * VEC,)), NB)
            elif MODE == 4:
                m = (bi & 1) == 0
                tl.atomic_add(hrow + bi, ones, mask=m, sem="relaxed", scope="cta")
            elif MODE == 5:
                flat = tl.reshape(bi, (BLOCK * VEC,))
                acc += tl.histogram(flat, NB, mask=(flat & 1) == 0)
            else:
                take = (offs % 64) == 0
                pos = tl.atomic_add(
                    chk_ptr + row + offs * 0,
                    ones,
                    mask=take,
                    sem="relaxed",
                    scope="cta",
                )
                tl.store(
                    out_ptr + row * K + pos, offs.to(tl.int32), mask=take & (pos < K)
                )
    if (MODE == 3) | (MODE == 5):
        tl.store(hrow + bins, acc)
    elif (MODE == 0) | (MODE == 1):
        tl.store(chk_ptr + row, chk.to(tl.int32))


MODES = (
    "load",
    "extract",
    "atomic",
    "histogram",
    "atomic_masked",
    "histogram_masked",
    "pass2_sparse",
)


def timed(fn, iters=10, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1000 / (ROWS / SMS)  # us per program


def main():
    dev = "cuda"
    torch.manual_seed(0)
    logits = torch.randn(ROWS * VOCAB, dtype=torch.float32, device=dev)
    hist = torch.zeros(ROWS * NB, dtype=torch.int32, device=dev)
    out = torch.zeros(ROWS * K, dtype=torch.int32, device=dev)
    chk = torch.zeros(ROWS, dtype=torch.int32, device=dev)
    kw = dict(VOCAB=VOCAB, BLOCK=BLOCK, VEC=VEC, NB=NB, K=K, num_warps=WARPS)
    print(
        f"{ROWS} programs = 10 waves on {SMS} SMs | {VOCAB} elements/program in "
        f"[{BLOCK},{VEC}] tiles | {NB} bins\n"
    )

    t, saved = {}, {}
    for mode, name in enumerate(MODES):
        try:
            t[name] = timed(
                lambda m=mode: k_hist[(ROWS,)](logits, hist, out, chk, MODE=m, **kw)
            )
            if name in ("atomic", "histogram", "atomic_masked", "histogram_masked"):
                saved[name] = hist.clone()
        except Exception as e:  # noqa: BLE001
            t[name] = None
            print(
                f"  {name}: FAILED {type(e).__name__}: {str(e).strip().splitlines()[-1][:120]}"
            )

    floor = t.get("load") or 0.0
    print(f"  {'mode':<18} {'us/prog':>9} {'minus load':>11} {'ns/element':>11}")
    for name in MODES:
        if t.get(name) is None:
            continue
        d = t[name] - floor
        print(f"  {name:<18} {t[name]:>9.2f} {d:>11.2f} {d * 1000 / VOCAB:>11.3f}")

    print()
    for a, b in (("atomic", "histogram"), ("atomic_masked", "histogram_masked")):
        if a in saved and b in saved:
            same = bool(torch.equal(saved[a], saved[b]))
            ta, tb = t[a] - t["extract"], t[b] - t["extract"]
            sp = ta / tb if tb > 0 else float("inf")
            print(
                f"  {a} vs {b}: counts {'IDENTICAL' if same else 'DIFFER'}; "
                f"histogram's own cost {ta:.2f} vs {tb:.2f} us -> {sp:.2f}x"
            )
    print("\n  Reference: the operator's whole vocab term is ~1.3-1.9 ns/element.")
    print("  'extract' is what every pass over the row pays; the operator makes")
    print("  at least two passes per step (histogram, then _process_bins).")


if __name__ == "__main__":
    sys.exit(main())
