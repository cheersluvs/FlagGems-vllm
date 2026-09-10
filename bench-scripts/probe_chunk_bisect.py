"""Bisect the minimal reproducer: which ingredient makes chunking corrupt Q?

WHERE THIS STANDS. A standalone kernel -- no KV arm, no branch, no fp8 encoder,
Q programs only -- reproduces the defect at the same magnitude as the shipped
one: 2043 elements differing from the single-launch answer and 2044 between
chunked runs, always the first program of a launch, always dims 448..511.
Ladder rung 5 had the same store-and-reload shape and stayed clean, so the
cause is among the few things the reproducer adds:

    bf16 storage    tl.split / tl.join    the RMS reduction over axis 1

Each variant below removes exactly one and changes nothing else. The variant
that comes back clean names the ingredient. If none does, the cause is the
combination and the reproducer is already minimal.

A NOTE ON THE PREVIOUS RUN. Its verdict said inconclusive because the detail
print raised: `nonzero()` on a [8192, 64, 512] boolean wants 6 GiB. The
measurement had already succeeded -- the numbers were printed -- but the script
scored it as a failure. Reductions are taken before nonzero here, and the
verdict no longer depends on the diagnostic succeeding. A probe that discards
its own result on a formatting error is worse than one that prints less.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_chunk_bisect.py
"""

import os
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

W, V, HH = 512, 64, 32
TOKENS, TILES = 8192, 2
REPEATS = 3


@triton.jit
def v0_full(q, src, table, pid_offset, tiles,
            V: tl.constexpr, HH: tl.constexpr, W: tl.constexpr):
    """Everything: bf16, masked store, split/join, RMS reduction."""
    pid = tl.program_id(0).to(tl.int64) + pid_offset
    tok = pid // tiles
    rows = tok * (tiles * HH) + (pid % tiles) * HH + tl.arange(0, HH)
    col = tl.arange(0, W)
    blk = tl.load(q + rows[:, None] * W + col[None, :]).to(tl.float32)
    rs = tl.rsqrt(tl.sum(blk * blk, axis=1) / W + 1e-6)
    blk = blk * rs[:, None]
    tl.store(q + rows[:, None] * W + col[None, :],
             blk.to(tl.bfloat16), mask=col[None, :] < W - V)
    p = tl.load(src + tok)
    half = tl.arange(0, V // 2)
    c = tl.load(table + p * V + half)
    s = tl.load(table + p * V + V // 2 + half)
    po = (rows[:, None, None] * W + (W - V)
          + half[None, :, None] * 2 + tl.arange(0, 2)[None, None, :])
    pair = tl.load(q + po).to(tl.float32)
    e, o = tl.split(pair)
    e, o = e * rs[:, None], o * rs[:, None]
    tl.store(q + po, tl.join(e * c[None, :] - o * s[None, :],
                             e * s[None, :] + o * c[None, :]).to(tl.bfloat16))


@triton.jit
def v1_no_store(q, src, table, pid_offset, tiles,
                V: tl.constexpr, HH: tl.constexpr, W: tl.constexpr):
    """Drop the masked NoPE store. Nothing writes q before the pair load."""
    pid = tl.program_id(0).to(tl.int64) + pid_offset
    tok = pid // tiles
    rows = tok * (tiles * HH) + (pid % tiles) * HH + tl.arange(0, HH)
    col = tl.arange(0, W)
    blk = tl.load(q + rows[:, None] * W + col[None, :]).to(tl.float32)
    rs = tl.rsqrt(tl.sum(blk * blk, axis=1) / W + 1e-6)
    p = tl.load(src + tok)
    half = tl.arange(0, V // 2)
    c = tl.load(table + p * V + half)
    s = tl.load(table + p * V + V // 2 + half)
    po = (rows[:, None, None] * W + (W - V)
          + half[None, :, None] * 2 + tl.arange(0, 2)[None, None, :])
    pair = tl.load(q + po).to(tl.float32)
    e, o = tl.split(pair)
    e, o = e * rs[:, None], o * rs[:, None]
    tl.store(q + po, tl.join(e * c[None, :] - o * s[None, :],
                             e * s[None, :] + o * c[None, :]).to(tl.bfloat16))


@triton.jit
def v2_no_splitjoin(q, src, table, pid_offset, tiles,
                    V: tl.constexpr, HH: tl.constexpr, W: tl.constexpr):
    """Drop split/join. The 3-D pair load and store stay; no pair axis work."""
    pid = tl.program_id(0).to(tl.int64) + pid_offset
    tok = pid // tiles
    rows = tok * (tiles * HH) + (pid % tiles) * HH + tl.arange(0, HH)
    col = tl.arange(0, W)
    blk = tl.load(q + rows[:, None] * W + col[None, :]).to(tl.float32)
    rs = tl.rsqrt(tl.sum(blk * blk, axis=1) / W + 1e-6)
    blk = blk * rs[:, None]
    tl.store(q + rows[:, None] * W + col[None, :],
             blk.to(tl.bfloat16), mask=col[None, :] < W - V)
    p = tl.load(src + tok)
    half = tl.arange(0, V // 2)
    c = tl.load(table + p * V + half)
    po = (rows[:, None, None] * W + (W - V)
          + half[None, :, None] * 2 + tl.arange(0, 2)[None, None, :])
    pair = tl.load(q + po).to(tl.float32)
    tl.store(q + po, (pair * rs[:, None, None] * c[None, :, None]).to(tl.bfloat16))


@triton.jit
def v3_no_rms(q, src, table, pid_offset, tiles,
              V: tl.constexpr, HH: tl.constexpr, W: tl.constexpr):
    """Drop the reduction. A constant scale replaces rsqrt(sum(...))."""
    pid = tl.program_id(0).to(tl.int64) + pid_offset
    tok = pid // tiles
    rows = tok * (tiles * HH) + (pid % tiles) * HH + tl.arange(0, HH)
    col = tl.arange(0, W)
    blk = tl.load(q + rows[:, None] * W + col[None, :]).to(tl.float32)
    rs = tl.full((HH,), 0.5, tl.float32)
    blk = blk * rs[:, None]
    tl.store(q + rows[:, None] * W + col[None, :],
             blk.to(tl.bfloat16), mask=col[None, :] < W - V)
    p = tl.load(src + tok)
    half = tl.arange(0, V // 2)
    c = tl.load(table + p * V + half)
    s = tl.load(table + p * V + V // 2 + half)
    po = (rows[:, None, None] * W + (W - V)
          + half[None, :, None] * 2 + tl.arange(0, 2)[None, None, :])
    pair = tl.load(q + po).to(tl.float32)
    e, o = tl.split(pair)
    e, o = e * rs[:, None], o * rs[:, None]
    tl.store(q + po, tl.join(e * c[None, :] - o * s[None, :],
                             e * s[None, :] + o * c[None, :]).to(tl.bfloat16))


VARIANTS = (("v0 everything (expect BROKEN)", v0_full, torch.bfloat16),
            ("v1 without the masked store", v1_no_store, torch.bfloat16),
            ("v2 without split/join", v2_no_splitjoin, torch.bfloat16),
            ("v3 without the RMS reduction", v3_no_rms, torch.bfloat16),
            ("v4 everything, but q is fp32", v0_full, torch.float32))


def main():
    import flaggems_vllm

    dev = flaggems_vllm.device
    sync = flaggems_vllm.runtime.torch_device_fn.synchronize
    torch.manual_seed(0)
    rows_total = TOKENS * TILES * HH
    src = torch.arange(TOKENS, dtype=torch.int64, device=dev)
    table = torch.randn(TOKENS, V, dtype=torch.float32, device=dev)
    q_programs = TOKENS * TILES
    CAP = q_programs // 2 + 1
    print("q_programs={}  cap={} -> launch 2 starts at token {}, tile {}"
          .format(q_programs, CAP, CAP // TILES, CAP % TILES))
    print("\n  {:<32} {:>10} {:>12}   verdict".format("variant", "vs single", "3 vs 3"))
    print("  " + "-" * 70)

    clean_ones = []
    for label, fn, dtype in VARIANTS:
        q0 = torch.randn(rows_total, W, dtype=dtype, device=dev)

        def run(cap):
            q = q0.clone()
            for off in range(0, q_programs, cap):
                fn[(min(cap, q_programs - off),)](
                    q, src, table, off, TILES, V, HH, W,
                    num_warps=1, num_stages=1)
            sync()
            return q

        try:
            ref = run(1 << 30)
            runs = [run(CAP) for _ in range(REPEATS)]
            vs_ref = int((ref != runs[0]).sum())
            vs_self = max(int((runs[0] != r).sum()) for r in runs[1:])
            clean = vs_ref == 0 and vs_self == 0
            print("  {:<32} {:>10} {:>12}   {}".format(
                label, vs_ref, "max {}".format(vs_self),
                "OK" if clean else "BROKEN"))
            if not clean:
                # reduce BEFORE nonzero; the full nonzero wants gigabytes
                dmask = (ref != runs[0])
                bad_rows = dmask.any(1).nonzero().flatten()
                bad_cols = dmask.any(0).nonzero().flatten()
                print("      rows {}..{} ({} of {})   cols {}..{}".format(
                    int(bad_rows.min()), int(bad_rows.max()),
                    int(bad_rows.numel()), rows_total,
                    int(bad_cols.min()), int(bad_cols.max())))
            else:
                clean_ones.append(label)
            del ref, runs
        except Exception as e:
            print("  {:<32} {:>10}   failed: {}".format(
                label, "-", str(e).splitlines()[0][:40]))
        del q0
        flaggems_vllm.runtime.torch_device_fn.empty_cache()

    print()
    if clean_ones:
        print("CLEAN WITHOUT: {}".format("; ".join(clean_ones)))
        print("Removing that is what stops the corruption, so it is the")
        print("ingredient. Confirm by putting it back alone on top of a cleared")
        print("variant before rewriting the shipped kernel around it.")
        print("\n[RESULT] INGREDIENT_FOUND")
    else:
        print("EVERY VARIANT IS STILL BROKEN. No single ingredient is")
        print("responsible -- v0 is already minimal in the dimensions tested, so")
        print("the next cut is structural: the [32, 512] fp32 tile is 64 KB")
        print("against roughly 36 KB of usable Unified Buffer, so try shrinking")
        print("HH and W before looking for another operation to blame.")
        print("\n[RESULT] NO_SINGLE_INGREDIENT")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
