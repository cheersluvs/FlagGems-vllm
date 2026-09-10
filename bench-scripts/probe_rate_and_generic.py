"""Failure RATE, measured honestly -- and is the generic kernel affected too?

TWO CORRECTIONS THIS MAKES.

1. THERE IS NO THRESHOLD, and the previous probe's verdict said otherwise. The
   sweep gave 2/5 at total 50, 5/5 at 57, 1/5 at 60 with 128 heads and 0/5 at 60
   with 64, 1/5 at 63, 0/5 at 70. Broken and clean interleave and the same total
   lands on both sides, so this is a shape-dependent PROBABILITY, not a size
   limit. Nothing here should be described as a threshold again.

2. EVERY `0/5` IN THIS INVESTIGATION IS UNPROVEN. With rates of 1/5 and 2/5
   actually observed, a shape whose true rate is 10% shows 0 in 5 runs about 59%
   of the time. So five runs cannot establish cleanliness, and three runs -- what
   backed the earlier claim that 1024 through 65536 are self-consistent -- are
   weaker still. Those shapes are not known to be clean; they were sampled too
   few times to say. This uses thirty.

THE QUESTION THAT DECIDES OWNERSHIP. The vendor override and the generic Triton
kernel compute the same operator; `flaggems_vllm.ops.<name>` reaches the generic
one directly, bypassing the binding the override replaces. Same shapes, same
inputs, same run count:

  only the override is non-deterministic -> the defect is in this repository's
                                            Ascend code and is ours to fix
  both are                                -> it is below both of them, in the
                                            compiler or the runtime, and the
                                            thing to produce is a vendor report,
                                            not another edit here

That is worth more than another attempt at the mechanism. Six have been proposed
and measured false, and the last two edits written against them -- a reorder and
a 2-D grid -- neither fixed this.

Reported as counts out of thirty, per shape, for both implementations, plus
where the differences land.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_rate_and_generic.py
"""

import os
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

HEAD_DIM, ROPE_DIM, HEAD_BYTES, EPS = 512, 64, 584, 1e-6
RUNS = 30
SHAPES = ((17, 64), (19, 64), (10, 128), (12, 128), (20, 64), (64, 64),
          (1024, 64), (8192, 64))


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
    import flaggems_vllm.ops as ops

    fn = flaggems_vllm.runtime.torch_device_fn
    dev = flaggems_vllm.device
    name = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    override = getattr(flaggems_vllm, name)
    generic = getattr(ops, name)
    same = override is generic
    print("override is generic: {}{}".format(
        same, "   <- no override is bound; the columns will be identical"
        if same else ""))

    def rate(impl, n, h):
        inp = build(n, h, dev)
        q, kv, kc, slot, pos, cs, b = inp

        def once():
            qq, kk = q.clone(), kc.clone()
            impl(qq, kv, kk, slot, pos, cs, EPS, b)
            fn.synchronize()
            return qq

        ref = once()
        bad, worst, where = 0, 0, ""
        for _ in range(RUNS):
            g = once()
            m = g != ref
            d = int(m.sum())
            if d:
                bad += 1
                if d > worst:
                    worst = d
                    tk = m.any(2).any(1).nonzero().flatten()
                    hd = m.any(2).any(0).nonzero().flatten()
                    where = "tok {}..{} head {}..{}".format(
                        int(tk.min()), int(tk.max()),
                        int(hd.min()), int(hd.max()))
            del g, m
        del ref, inp
        fn.empty_cache()
        return bad, worst, where

    print("\n  {:>7} {:>5} {:>7} {:>14} {:>14} {:>22}".format(
        "tokens", "heads", "total", "override /30", "generic /30", "where (override)"))
    print("  " + "-" * 74)
    ov_bad, gen_bad = [], []
    for n, h in SHAPES:
        tiles = h // 32
        tot = n * tiles + n
        try:
            ob, ow, wh = rate(override, n, h)
            gb, gw, _ = (ob, ow, wh) if same else rate(generic, n, h)
            ov_bad.append((n, h, ob))
            gen_bad.append((n, h, gb))
            print("  {:>7} {:>5} {:>7} {:>14} {:>14} {:>22}".format(
                n, h, tot, "{}/{}".format(ob, RUNS), "{}/{}".format(gb, RUNS), wh))
        except Exception as e:
            print("  {:>7} {:>5} {:>7}   {}".format(
                n, h, tot, str(e).splitlines()[0][:44]))

    o_any = any(b for _, _, b in ov_bad)
    g_any = any(b for _, _, b in gen_bad)
    print()
    if same:
        print("NO OVERRIDE WAS BOUND, so this compared one implementation with")
        print("itself and says nothing about ownership. Check the vendor")
        print("registration before reading anything into the columns.")
        print("\n[RESULT] NO_OVERRIDE_BOUND")
    elif o_any and g_any:
        print("BOTH IMPLEMENTATIONS ARE NON-DETERMINISTIC. The defect is below")
        print("both of them -- in the compiler or the runtime -- so the output")
        print("owed here is a vendor report with this reproducer, not another")
        print("edit to the override. Note that the generic kernel ships on every")
        print("backend, so the same code is exposed elsewhere.")
        print("\n[RESULT] BOTH_AFFECTED")
    elif o_any:
        print("ONLY THE OVERRIDE IS NON-DETERMINISTIC; the generic kernel is")
        print("stable over {} runs at the same shapes. The defect is in this")
        print("repository's Ascend code.".format(RUNS))
        print("\n[RESULT] OVERRIDE_ONLY")
    else:
        print("NEITHER REPRODUCED in {} runs. Given 19-of-19 at 17x64 earlier,")
        print("that is a change in behaviour, not evidence of correctness --")
        print("something about this session differs and it needs finding before")
        print("any of this is written up.".format(RUNS))
        print("\n[RESULT] NOTHING_REPRODUCED")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
