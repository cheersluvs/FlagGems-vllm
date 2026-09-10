"""Can a 2-D grid replace the chunked launches? That would retire the defect.

WHY STOP HUNTING THE ROOT CAUSE. Six mechanisms have been proposed and measured
false: memory, the rotation arithmetic, the RMS scale, ten input substitutions,
pid_offset arithmetic with a scalar gather and a broadcast, the masked store
before the reload -- for which a pure reorder was written and refuted -- and the
launch configuration. The standalone reproducer then turned out to be unstable
as a single launch, so it reproduces a neighbouring bug and its bisection is
void. What is solid is narrow and useful: exactly one program per launch comes
out non-deterministically wrong, and the shipped kernel is clean and repeatable
when it issues ONE launch.

So the cheapest safe move is not to fix the chunking but to stop needing it.

WHAT THE LIMIT ACTUALLY SAID. The runtime rejected a launch with

    value 532480 for parameter coreDim is invalid.
    Expected value: less than or equal to 65535

`coreDim` reads like the FIRST grid dimension, not the total program count. If
that is right, a 2-D grid of (65535, ceil(total / 65535)) launches the same work
in one go, `pid_offset` and its loop disappear, and the only path that has ever
measured clean is the only path left. If coreDim is instead the total, a 2-D
grid fails the same way and this whole direction is closed -- worth one run to
find out rather than an assumption either way.

WHAT IS TESTED, in order, because a later answer is meaningless if an earlier
one fails:

  1. does a 2-D grid launch at all, at the real program count for
     131072 tokens x 128 heads -- the largest shape in the benchmark
  2. does pid reconstructed as program_id(0) + program_id(1) * dim0 cover every
     index exactly once, with no gap and no duplicate
  3. is the result repeatable across five runs, since a launch that succeeds
     and then computes non-deterministically is no better than what it replaces

Only all three passing makes this a candidate. Nothing here touches the shipped
kernel; the change it would justify is a separate edit, measured separately.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_grid2d.py
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

LIMIT = 65535


@triton.jit
def write_pid_2d(out, n, dim0):
    pid = tl.program_id(0).to(tl.int64) + tl.program_id(1).to(tl.int64) * dim0
    if pid < n:
        tl.store(out + pid, pid)


def main():
    import flaggems_vllm

    dev = flaggems_vllm.device
    sync = flaggems_vllm.runtime.torch_device_fn.synchronize

    # the largest shape the benchmark uses: 131072 tokens, 128 heads, H=32
    tiles = 128 // 32
    cases = [("8192 x 64", 8192 * 2 + 8192),
             ("131072 x 64", 131072 * 2 + 131072),
             ("131072 x 128", 131072 * tiles + 131072)]

    print("  {:<16} {:>12} {:>8} {:>8} {:>12} {:>10}".format(
        "shape", "programs", "dim0", "dim1", "coverage", "5 runs"))
    print("  " + "-" * 72)

    ok_all = True
    for label, total in cases:
        dim0 = min(LIMIT, total)
        dim1 = -(-total // dim0)
        try:
            outs = []
            for _ in range(5):
                out = torch.full((total,), -1, dtype=torch.int64, device=dev)
                write_pid_2d[(dim0, dim1)](out, total, dim0, num_warps=1,
                                           num_stages=1)
                sync()
                outs.append(out)
            want = torch.arange(total, dtype=torch.int64, device=dev)
            covered = bool(torch.equal(outs[0], want))
            stable = all(torch.equal(outs[0], o) for o in outs[1:])
            missing = int((outs[0] < 0).sum())
            print("  {:<16} {:>12} {:>8} {:>8} {:>12} {:>10}".format(
                label, total, dim0, dim1,
                "exact" if covered else "{} missing".format(missing),
                "identical" if stable else "VARIES"))
            ok_all = ok_all and covered and stable
            del outs, want
            flaggems_vllm.runtime.torch_device_fn.empty_cache()
        except Exception as e:
            print("  {:<16} {:>12} {:>8} {:>8}   failed: {}".format(
                label, total, dim0, dim1, str(e).splitlines()[0][:40]))
            ok_all = False

    print()
    if ok_all:
        print("A 2-D GRID WORKS. coreDim limits the first dimension only, every")
        print("index is covered exactly once, and the mapping is repeatable at")
        print("the largest benchmark shape. So the chunking loop can go: one")
        print("launch, no pid_offset, and the only path that has measured clean")
        print("becomes the only path there is.")
        print("\nNext, and separately: make that edit, then re-run")
        print("probe_chunked_launch.py and probe_chunk_repeat.py plus the full")
        print("test suite, since a launch that succeeds is not yet a launch that")
        print("computes the right answer.")
        print("\n[RESULT] GRID2D_VIABLE")
    else:
        print("A 2-D GRID DOES NOT GIVE A CLEAN COVER HERE, so this direction is")
        print("closed and the chunking loop has to stay. Back to locating the")
        print("defect -- and the next thing to vary is the branch, which the")
        print("shipped kernel has and the unstable standalone copy did not.")
        print("\n[RESULT] GRID2D_CLOSED")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
