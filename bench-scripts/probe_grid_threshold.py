"""Is the small-grid defect about grid size, or about head count? Pin it.

WHAT IS ESTABLISHED. Program 0 -- token 0, heads 0..31, the first program of
the grid -- computes non-deterministically at small grids: 19 of 19 runs
differed at 17 tokens x 64 heads, in both NoPE and RoPE. Sweeping token counts
gave a clean break at 64 heads, broken at 15..19 tokens and clean from 20, and
every 128-head case clean.

WHY THAT LAST PART MAY MEAN NOTHING. total_programs is 3n at 64 heads and 5n at
128, so the 128-head column never went below 75 while the break at 64 heads sits
between 57 and 60. The head count and the grid size are confounded, exactly as
the failing shapes and the multi-launch shapes were confounded earlier -- and
that confusion cost several rounds. So the 128-head column is tested BELOW the
threshold this time, which the previous sweep never did.

  head count matters      -> 128 heads stays clean at total 40 and 50
  grid size matters       -> 128 heads breaks there too, and the head count is
                             a coincidence of how total_programs is computed

WHICH QUANTITY, if it is size. q_programs and total_programs are distinguished
by construction: 64 heads at 20 tokens is q=40, total=60 and clean; 128 heads at
10 tokens is q=40, total=50. If that one is clean, the threshold is q_programs;
if it breaks, it is total_programs. A third possibility, that it tracks the
device's core count, is why the properties are printed.

Each row is five runs against a reference, all on one input, one reference and
one run live at a time.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_grid_threshold.py
"""

import os
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

HEAD_DIM, ROPE_DIM, HEAD_BYTES, EPS = 512, 64, 584, 1e-6


def build(n, h, dev, b=64):
    torch.manual_seed(0)
    q = torch.randn(n, h, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(n, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    pos = torch.arange(n, dtype=torch.int64, device=dev)
    inv = 1.0 / (10000.0 ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32,
                                          device=dev) / ROPE_DIM))
    t = torch.arange(max(4096, n), dtype=torch.float32, device=dev)
    fr = torch.einsum("i,j->ij", t, inv)
    cs = torch.cat((fr.cos(), fr.sin()), dim=-1)
    nb = (n + b - 1) // b + 1
    slot = torch.arange(n, dtype=torch.int64, device=dev)
    kc = torch.zeros(nb, b * HEAD_BYTES, dtype=torch.uint8, device=dev)
    return q, kv, kc, slot, pos, cs, b


def main():
    import flaggems_vllm

    impl = flaggems_vllm.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    dev = flaggems_vllm.device
    fn = flaggems_vllm.runtime.torch_device_fn
    try:
        p = fn.get_device_properties(0)
        bits = [a for a in ("name", "total_memory", "multi_processor_count",
                            "cube_core_num", "vector_core_num", "aicore_num")
                if hasattr(p, a)]
        print("device: " + "  ".join("{}={}".format(a, getattr(p, a)) for a in bits))
    except Exception as e:
        print("device properties unavailable: {}".format(str(e)[:60]))

    def check(n, h):
        inp = build(n, h, dev)
        q, kv, kc, slot, pos, cs, b = inp

        def once():
            qq, kk = q.clone(), kc.clone()
            impl(qq, kv, kk, slot, pos, cs, EPS, b)
            fn.synchronize()
            return qq

        ref = once()
        bad, worst = 0, 0
        for _ in range(5):
            g = once()
            d = int((g != ref).sum())
            bad += 1 if d else 0
            worst = max(worst, d)
            del g
        del ref, inp
        fn.empty_cache()
        return bad, worst

    print("\n" + "=" * 78)
    print("128 heads BELOW the threshold -- the case the last sweep never reached")
    print("=" * 78)
    print("  {:>7} {:>6} {:>9} {:>8} {:>12} {:>12}".format(
        "tokens", "heads", "q_prog", "total", "runs differ", "elements"))
    print("  " + "-" * 62)
    rows = []
    for n, h in ((4, 128), (6, 128), (8, 128), (10, 128), (12, 128),
                 (14, 128), (4, 64), (8, 64), (12, 64), (19, 64),
                 (20, 64), (21, 64), (24, 64)):
        tiles = h // 32
        qp, tot = n * tiles, n * tiles + n
        try:
            bad, worst = check(n, h)
            rows.append((n, h, qp, tot, bad))
            print("  {:>7} {:>6} {:>9} {:>8} {:>12} {:>12}".format(
                n, h, qp, tot, "{}/5".format(bad), worst))
        except Exception as e:
            print("  {:>7} {:>6} {:>9} {:>8}   {}".format(
                n, h, qp, tot, str(e).splitlines()[0][:36]))

    print("\n" + "=" * 78)
    broken = [r for r in rows if r[4] > 0]
    clean = [r for r in rows if r[4] == 0]
    h128_broken = [r for r in broken if r[1] == 128]
    if h128_broken:
        print("128 HEADS BREAKS TOO, below the threshold. The head count was a")
        print("coincidence of total_programs being 3n at 64 heads and 5n at 128,")
        print("so the earlier clean 128 column meant nothing -- it never went low")
        print("enough. GRID SIZE is the variable.")
        if broken and clean:
            print("\n  largest broken total: {}   smallest clean total: {}".format(
                max(r[3] for r in broken), min(r[3] for r in clean)))
            qb = max(r[2] for r in broken)
            qc = min(r[2] for r in clean)
            print("  largest broken q_programs: {}   smallest clean: {}".format(qb, qc))
            print("\n  Whichever of the two has no overlap is the quantity that")
            print("  matters; if both separate cleanly, the pair (10, 128) versus")
            print("  (20, 64) -- same q_programs, different total -- decides it.")
        print("\n[RESULT] GRID_SIZE")
    elif broken:
        print("128 heads stays clean even below the threshold, while 64 heads")
        print("breaks. So it is NOT grid size alone -- tiles_per_token, which is")
        print("2 at 64 heads and 4 at 128, is in the causal path.")
        print("\n[RESULT] HEAD_COUNT_MATTERS")
    else:
        print("Nothing broke in this run. For a flaky defect that is data, not")
        print("absence -- the 17x64 case reproduced 19 times out of 19, so a")
        print("clean sweep here is itself surprising and worth repeating before")
        print("it is believed.")
        print("\n[RESULT] NOTHING_REPRODUCED")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
