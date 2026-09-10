"""A ladder of minimal kernels: which single ingredient breaks under chunking?

WHY A LADDER. Five mechanisms have now been proposed for this defect and
measured false -- wrong position, missing scale, double rotation, swapped
cos/sin, and most recently the masked NoPE store preceding the RoPE re-load,
which a pure reorder failed to fix. Proposing a sixth is the same move. This
stops explaining the real kernel and instead builds up from a kernel that
cannot be wrong, one ingredient at a time, until chunking breaks it. The first
rung that breaks names the cause, with no theory required.

WHAT EACH RUNG ADDS, and what its failure would mean:

  1 pid            write back the global program id
                   -> breaks: pid_offset arithmetic itself is wrong. Nothing
                      about RoPE, q, cos/sin or masks is involved, and this is a
                      backend or Triton defect to report, not our code to fix.
  2 scalar load    load one int64 per token and write it back
                   -> breaks: a scalar gather indexed off pid is wrong
  3 table row      use that scalar to index a second table, load a vector
                   -> breaks: the two-level indirection is wrong
  4 broadcast      broadcast that vector across a head axis into a tile
                   -> breaks: the broadcast is what fails. This is the rung the
                      observed evidence points at, because the bad program's
                      angle DIFFERED BETWEEN HEADS while cos_blk[None, :] must
                      give every head the same values.
  5 store+reload   the real structure: masked store, then reload a disjoint
                   column range from the same tensor
                   -> breaks only here: the hazard really is the ordering, and
                      the reorder did not remove it

Every rung is checked the same way: run it as one launch, run it chunked, and
compare. One launch is the reference, so nothing external can enter the answer.
Each rung also runs chunked THREE times to catch the non-determinism that the
real kernel shows, since a rung that is merely usually-right must not be scored
as clean.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_chunk_ladder.py
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

H = 32
WIDTH = 512
VEC = 32
TOKENS = 4096
TILES = 2                      # programs per token, as at 64 heads
REPEATS = 3


@triton.jit
def rung1(out, pid_offset):
    pid = tl.program_id(0).to(tl.int64) + pid_offset
    tl.store(out + pid, pid)


@triton.jit
def rung2(out, src, pid_offset, tiles):
    pid = tl.program_id(0).to(tl.int64) + pid_offset
    tok = pid // tiles
    tl.store(out + pid, tl.load(src + tok))


@triton.jit
def rung3(out, src, table, pid_offset, tiles, V: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64) + pid_offset
    tok = pid // tiles
    p = tl.load(src + tok)
    v = tl.load(table + p * V + tl.arange(0, V))
    tl.store(out + pid * V + tl.arange(0, V), v)


@triton.jit
def rung4(out, src, table, pid_offset, tiles, V: tl.constexpr, HH: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64) + pid_offset
    tok = pid // tiles
    p = tl.load(src + tok)
    v = tl.load(table + p * V + tl.arange(0, V))
    rowscale = (tl.arange(0, HH) + 1).to(tl.float32)
    tile = v[None, :] * rowscale[:, None]
    off = pid * (HH * V) + tl.arange(0, HH)[:, None] * V + tl.arange(0, V)[None, :]
    tl.store(out + off, tile)


@triton.jit
def rung5(q, out, src, table, pid_offset, tiles,
          V: tl.constexpr, HH: tl.constexpr, W: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64) + pid_offset
    tok = pid // tiles
    rows = tok * (tiles * HH) + (pid % tiles) * HH + tl.arange(0, HH)
    col = tl.arange(0, W)
    blk = tl.load(q + rows[:, None] * W + col[None, :]).to(tl.float32)
    tl.store(q + rows[:, None] * W + col[None, :],
             (blk * 2.0).to(tl.float32), mask=col[None, :] < W - V)
    p = tl.load(src + tok)
    v = tl.load(table + p * V + tl.arange(0, V))
    tail_off = rows[:, None] * W + (W - V) + tl.arange(0, V)[None, :]
    tail = tl.load(q + tail_off).to(tl.float32)
    tl.store(out + pid * (HH * V) + tl.arange(0, HH)[:, None] * V
             + tl.arange(0, V)[None, :], tail * v[None, :])


def main():
    import flaggems_vllm

    dev = flaggems_vllm.device
    sync = flaggems_vllm.runtime.torch_device_fn.synchronize
    total = TOKENS * TILES

    src = torch.arange(TOKENS, dtype=torch.int64, device=dev)
    table = torch.randn(TOKENS * 8, VEC, dtype=torch.float32, device=dev)
    q0 = torch.randn(TOKENS * TILES * H, WIDTH, dtype=torch.float32, device=dev)

    def launch(fn, out, caps, extra, q=None):
        for off in range(0, total, caps):
            grid = min(caps, total - off)
            if q is None:
                fn[(grid,)](out, *extra[:0], **{}) if False else None
            # explicit per-rung dispatch below; this helper only loops
        return out

    def go(rung, cap):
        if rung == 1:
            out = torch.zeros(total, dtype=torch.int64, device=dev)
            for off in range(0, total, cap):
                rung1[(min(cap, total - off),)](out, off)
        elif rung == 2:
            out = torch.zeros(total, dtype=torch.int64, device=dev)
            for off in range(0, total, cap):
                rung2[(min(cap, total - off),)](out, src, off, TILES)
        elif rung == 3:
            out = torch.zeros(total * VEC, dtype=torch.float32, device=dev)
            for off in range(0, total, cap):
                rung3[(min(cap, total - off),)](out, src, table, off, TILES, VEC)
        elif rung == 4:
            out = torch.zeros(total * H * VEC, dtype=torch.float32, device=dev)
            for off in range(0, total, cap):
                rung4[(min(cap, total - off),)](out, src, table, off, TILES, VEC, H)
        else:
            out = torch.zeros(total * H * VEC, dtype=torch.float32, device=dev)
            q = q0.clone()
            for off in range(0, total, cap):
                rung5[(min(cap, total - off),)](q, out, src, table, off, TILES,
                                                VEC, H, WIDTH)
        sync()
        return out

    names = {1: "pid", 2: "scalar load", 3: "table row", 4: "broadcast",
             5: "store+reload"}
    cap = total // 2 + 1            # exactly one boundary, inside a token
    print("=" * 74)
    print("  {} tokens x {} tiles = {} programs; chunk cap {} -> 2 launches"
          .format(TOKENS, TILES, total, cap))
    print("  boundary program {} is token {}, tile {}"
          .format(cap, cap // TILES, cap % TILES))
    print("=" * 74)
    print("\n  {:<16} {:>12} {:>34}".format("rung", "single vs chunked", "3 chunked runs vs each other"))
    print("  " + "-" * 66)

    first_break = None
    for r in (1, 2, 3, 4, 5):
        try:
            ref = go(r, 1 << 30)
            runs = [go(r, cap) for _ in range(REPEATS)]
            vs_ref = int((ref != runs[0]).sum())
            vs_self = max(int((runs[0] != x).sum()) for x in runs[1:])
            verdict = "OK" if vs_ref == 0 and vs_self == 0 else "BROKEN"
            print("  {:<16} {:>12} {:>34}   {}".format(
                "{}. {}".format(r, names[r]), vs_ref,
                "max {}".format(vs_self), verdict))
            if verdict == "BROKEN" and first_break is None:
                first_break = r
            del ref, runs
            flaggems_vllm.runtime.torch_device_fn.empty_cache()
        except Exception as e:
            print("  {:<16} {:>12}   did not compile/run: {}".format(
                "{}. {}".format(r, names[r]), "-",
                str(e).splitlines()[0][:70]))

    print()
    if first_break is None:
        print("EVERY RUNG IS CLEAN. None of these ingredients breaks under")
        print("chunking, so the cause is something the ladder does not yet")
        print("contain -- add the next difference from the real kernel (num_warps,")
        print("the fp8 encoder, the KV arm sharing the grid) rather than")
        print("re-explaining the ones already cleared.")
        print("\n[RESULT] LADDER_ALL_CLEAN")
    else:
        print("FIRST RUNG TO BREAK: {}. {}".format(first_break, names[first_break]))
        if first_break == 1:
            print("pid_offset arithmetic alone is wrong. No q, no mask, no RoPE,")
            print("no cos/sin. This is a backend defect, not this operator's code.")
        elif first_break == 4:
            print("The broadcast across the head axis is what fails, which matches")
            print("the angle differing between heads while cos_blk[None, :] must")
            print("give every head the same values.")
        print("\n[RESULT] LADDER_BREAKS_AT_{}".format(first_break))


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
