"""Is the standalone reproducer actually reproducing THIS defect? Baseline first.

WHY THE BISECTION IS NOT READABLE YET. It compared five variants against a
single-launch reference taken as ONE sample, and its v0 came back with the NoPE
region corrupted (cols 0..511) and row 0 among the bad rows. Neither is true of
the shipped kernel, where NoPE is always clean and only the first program of a
launch AFTER the first goes wrong. Two readings fit:

  * the standalone kernel is unstable even as a single launch, so every variant
    was measured against a moving target and the numbers 32765 / 4092 / 31333
    cannot be compared to each other at all
  * or it is stable and genuinely fails differently, meaning it reproduces a
    DIFFERENT defect and bisecting it says nothing about the shipped one

Both make the bisection void, and they are told apart by one measurement that
was omitted: run the single launch repeatedly.

THE DEFINING PROPERTY. What has to be reproduced is not "wrong output" but
LAUNCH-COUNT DEPENDENCE -- correct and repeatable as one launch, wrong and
unrepeatable when split. A copy that is simply always wrong is not a smaller
version of this bug; it is a different bug that happens to be in the
neighbourhood. That distinction is the whole value of a reproducer, so it is
checked before anything is bisected inside it.

Also measured: whether the [HH, W] fp32 tile is the variable. 32 x 512 in
float32 is 64 KB against roughly 36 KB of usable Unified Buffer on this part,
so the tile is over budget before the pair load is counted. Halving HH twice
says whether that matters, and it costs three runs to find out.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_repro_baseline.py
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

W, V = 512, 64
TOKENS = 8192
REPEATS = 5


@triton.jit
def q_arm(q, src, table, pid_offset, tiles,
          V: tl.constexpr, HH: tl.constexpr, W: tl.constexpr):
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


def main():
    import flaggems_vllm

    dev = flaggems_vllm.device
    sync = flaggems_vllm.runtime.torch_device_fn.synchronize

    print("  {:>4} {:>7} {:>22} {:>22} {:>10}".format(
        "HH", "tiles", "single launch x5", "chunked x5", "reads as"))
    print("  " + "-" * 72)

    verdicts = {}
    for HH in (32, 16, 8):
        tiles = 64 // HH
        rows_total = TOKENS * tiles * HH
        q_programs = TOKENS * tiles
        cap = q_programs // 2 + 1
        torch.manual_seed(0)
        q0 = torch.randn(rows_total, W, dtype=torch.bfloat16, device=dev)
        src = torch.arange(TOKENS, dtype=torch.int64, device=dev)
        table = torch.randn(TOKENS, V, dtype=torch.float32, device=dev)

        def run(c):
            q = q0.clone()
            for off in range(0, q_programs, c):
                q_arm[(min(c, q_programs - off),)](
                    q, src, table, off, tiles, V, HH, W,
                    num_warps=1, num_stages=1)
            sync()
            return q

        try:
            singles = [run(1 << 30) for _ in range(REPEATS)]
            s_bad = max(int((singles[0] != x).sum()) for x in singles[1:])
            chunked = [run(cap) for _ in range(REPEATS)]
            c_bad = max(int((chunked[0] != x).sum()) for x in chunked[1:])
            c_vs_s = int((singles[0] != chunked[0]).sum())

            if s_bad == 0 and c_bad == 0 and c_vs_s == 0:
                reads = "all clean"
            elif s_bad == 0 and (c_bad or c_vs_s):
                reads = "REPRODUCES"
            elif s_bad:
                reads = "unstable"
            else:
                reads = "?"
            verdicts[HH] = reads
            print("  {:>4} {:>7} {:>22} {:>22} {:>10}".format(
                HH, tiles, "max {}".format(s_bad),
                "max {} / vs single {}".format(c_bad, c_vs_s), reads))
            del singles, chunked
        except Exception as e:
            print("  {:>4} {:>7}   failed: {}".format(
                HH, tiles, str(e).splitlines()[0][:44]))
            verdicts[HH] = "failed"
        del q0
        flaggems_vllm.runtime.torch_device_fn.empty_cache()

    print()
    if verdicts.get(32) == "unstable":
        print("THE REPRODUCER IS UNSTABLE AS A SINGLE LAUNCH, so it does not")
        print("reproduce this defect -- the shipped kernel is repeatable when")
        print("not chunked. The bisection built on it is void; discard those")
        print("five numbers rather than reasoning from them.")
        v = "REPRO_INVALID"
    elif verdicts.get(32) == "REPRODUCES":
        smaller = [h for h in (16, 8) if verdicts.get(h) == "all clean"]
        print("THE REPRODUCER IS VALID at HH=32: clean and repeatable as one")
        print("launch, wrong when split. That is the defining property.")
        if smaller:
            print("And it goes away at HH={} -- the tile size IS the variable, so"
                  .format(smaller[0]))
            print("the fix direction is a smaller tile, not a different operation.")
            v = "TILE_SIZE_IS_THE_VARIABLE"
        else:
            print("It persists at HH=16 and HH=8, so the tile size is not the")
            print("variable and the 64 KB against 36 KB reading is wrong.")
            v = "NOT_TILE_SIZE"
    else:
        print("Read the rows above; the HH=32 case did not behave as either.")
        v = "UNCLEAR"
    print("\n[RESULT] {}".format(v))


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
