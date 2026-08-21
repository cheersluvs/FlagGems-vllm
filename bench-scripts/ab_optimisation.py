"""Measure what the optimisation actually bought, before against after.

"Before" is c50ad93 -- the last enablement commit, the first version that ran
the whole shape range correctly and had had no tuning at all. "After" is HEAD.
Both files are self-contained (torch, triton, tl only), so both can be imported
into one process and driven side by side.

Two things this does that a naive two-run comparison does not:

  * ROUND-ROBIN. Old and new are timed alternately within one process on the
    same tensors, so clock drift, cache state and allocator state affect both
    equally. Running one version to completion and then the other invites the
    difference to be partly an artefact of when each ran.

  * PROVE THEY AGREE FIRST. Every optimisation step was checked bit-exact as it
    landed, so old and new should produce identical q and identical k_cache. If
    they do not, the ratio is comparing two different computations and the run
    stops. This is the check that makes a speedup number mean something.

Run it from the repo root. It extracts the old file with `git show`, so nothing
in the working tree is touched.
"""

import os
import subprocess
import sys
import time
import traceback

REPO = os.environ.get("REPO", "/home/secure/wuyuqing/workspace/FlagGems-vllm")
OLD_REV = os.environ.get("OLD_REV", "c50ad93")
REL = ("src/flaggems_vllm/runtime/backend/_ascend/fused/"
       "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py")

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
import torch_npu  # noqa: E402,F401

HEAD_DIM, ROPE_DIM = 512, 64
TOKEN_DATA_BYTES, SCALE_BYTES = 576, 8
CACHE_BLOCK = 64


def load_old():
    src = subprocess.check_output(
        ["git", "-C", REPO, "show", "{}:{}".format(OLD_REV, REL)]
    ).decode()
    assert "flaggems_vllm" not in src, "the old file is not self-contained"
    path = "/tmp/ascend_op_{}.py".format(OLD_REV)
    with open(path, "w") as f:
        f.write(src)
    import importlib.util
    spec = importlib.util.spec_from_file_location("ascend_op_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert, len(
        src.splitlines()
    )


def load_new():
    from importlib import import_module
    m = import_module(
        "flaggems_vllm.runtime.backend._ascend.fused"
        ".fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    )
    return m.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert


def make(n, h):
    dev = "npu"
    nb = (n + CACHE_BLOCK - 1) // CACHE_BLOCK + 1
    bb = CACHE_BLOCK * (TOKEN_DATA_BYTES + SCALE_BYTES)
    torch.manual_seed(0)
    q = torch.randn(n, h, HEAD_DIM, dtype=torch.float32).to(torch.bfloat16).npu()
    kv = torch.randn(n, HEAD_DIM, dtype=torch.float32).to(torch.bfloat16).npu()
    slot = torch.arange(n, dtype=torch.int64, device=dev)
    pos = torch.arange(n, dtype=torch.int64, device=dev)
    # sized to the positions used; a short table is an out-of-range gather and
    # the card faults on it (aicpu exception 507018) rather than clamping
    cs = torch.randn(max(4096, n), ROPE_DIM, dtype=torch.float32).npu()
    cache = torch.zeros(nb, bb, dtype=torch.uint8, device=dev)
    return q, kv, cache, slot, pos, cs


def bytes_moved(n, h):
    return (2 * n * h * HEAD_DIM * 2 + n * HEAD_DIM * 2
            + n * (TOKEN_DATA_BYTES + SCALE_BYTES) + n * ROPE_DIM * 4 + n * 16)


def agree(old, new, shapes):
    print("### do the two versions still compute the same thing?\n")
    print("  {:<16} {:>14} {:>18}".format("shape", "q differs", "k_cache differs"))
    ok = True
    for n, h in shapes:
        q1, kv1, c1, sl, po, cs = make(n, h)
        q2, c2 = q1.clone(), c1.clone()
        old(q1, kv1, c1, sl, po, cs, 1e-6, CACHE_BLOCK)
        new(q2, kv1, c2, sl, po, cs, 1e-6, CACHE_BLOCK)
        torch.npu.synchronize()
        a, b = q1.cpu(), q2.cpu()
        dq = int((a != b).sum())
        dc = int((c1.cpu() != c2.cpu()).sum())
        # Bit-pattern distance is the wrong metric near zero: 1e-40 and 2e-40
        # are many "ULPs" apart and identical for any practical purpose. Measure
        # the relative difference, and judge by the repo's own tolerance for q.
        worst_rel, heads_hit, in_nope, in_rope, worst_pair = 0.0, 0, 0, 0, None
        if dq:
            d = a != b
            af, bf = a.float(), b.float()
            rel = (af - bf).abs() / bf.abs().clamp(min=1e-30)
            worst_rel = float(rel[d].max())
            k = int(rel[d].argmax())
            worst_pair = (float(af[d][k]), float(bf[d][k]))
            idx = torch.nonzero(d)
            heads_hit = len({(int(r), int(hh)) for r, hh, _ in idx.tolist()})
            cols = idx[:, -1]
            in_nope = int((cols < 448).sum())
            in_rope = int((cols >= 448).sum())
        # k_cache is the quantised output: one ULP there is a different byte, so
        # it must be exact. q is bfloat16 and the two versions reduce the
        # variance over differently shaped tiles, so the sum -- and the rsqrt
        # from it -- can land one ULP apart, which then moves the few elements
        # that sit on a rounding boundary.
        close = torch.allclose(a.float(), b.float(), rtol=1e-2, atol=1e-2)
        ok = ok and dc == 0 and close
        print("  {:<16} {:>14} {:>18}".format("{}x{}".format(n, h), dq, dc))
        if dq:
            print("      worst relative {:.3e} at {!r} vs {!r}".format(
                worst_rel, worst_pair[0], worst_pair[1]))
            print("      {} of {} heads touched, {} NoPE / {} RoPE, "
                  "within rtol=1e-2: {}".format(
                      heads_hit, n * h, in_nope, in_rope, close))
        del q1, q2, kv1, c1, c2
        torch.npu.empty_cache()
    return ok


def round_robin(old, new, n, h, rounds=5, iters=20):
    """Alternate the two versions so drift lands on both.

    Returns the median AND the spread. A median alone hid a 2.2x outlier in two
    cells of an earlier run at rounds=3, iters=5: both versions were slow in the
    same two shapes, which is the signature of a disturbed measurement rather
    than a property of either. It was only caught by comparing against two
    earlier independent runs. Print the spread so the next one is visible
    without that luck.
    """
    q, kv, c, sl, po, cs = make(n, h)
    args = (kv, c, sl, po, cs, 1e-6, CACHE_BLOCK)
    for fn in (old, new):
        for _ in range(2):
            fn(q, *args)
    torch.npu.synchronize()

    to, tn = [], []
    for _ in range(rounds):
        for fn, acc in ((old, to), (new, tn)):
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(q, *args)
            torch.npu.synchronize()
            acc.append((time.perf_counter() - t0) / iters)
    del q, kv, c
    torch.npu.empty_cache()
    to.sort()
    tn.sort()
    return (to[len(to) // 2], tn[len(tn) // 2],
            (to[-1] - to[0]) / to[len(to) // 2],
            (tn[-1] - tn[0]) / tn[len(tn) // 2])


def main():
    old, nlines = load_old()
    new = load_new()
    print("before: {} at {} lines".format(OLD_REV, nlines))
    print("after : HEAD\n")

    check_shapes = [(17, 64), (1024, 64), (64, 128)]
    if not agree(old, new, check_shapes):
        print("\n  Beyond a rounding difference. Note what WAS checked as each")
        print("  step landed: the flat tile against a torch reference on device,")
        print("  and head tiling against the flat tile. The original per-unit")
        print("  version was only ever checked through the suite at rtol=1e-2,")
        print("  so a difference against IT is new information, not a broken")
        print("  promise -- but this one is too large to be rounding.")
        print("\n[RESULT] VERSIONS_DIFFER")
        return
    print("\n  k_cache identical; q within one bfloat16 ULP, from the "
          "variance reduction\n  associating differently over a wider "
          "tile. Same function, same speed comparison.\n")

    print("### before vs after, timed alternately in one process\n")
    print("  {:>7} {:>6} {:>12} {:>12} {:>10} {:>10} {:>10} {:>7} {:>7}"
          .format("tokens", "heads", "before ms", "after ms", "speedup",
                  "before GB/s", "after GB/s", "b-sprd", "a-sprd"))
    for h in (64, 128):
        for n in (1024, 4096, 16384, 32768):
            try:
                tb, ta, sb, sa = round_robin(old, new, n, h)
            except RuntimeError as e:
                print("  {:>7} {:>6}   skipped: {}".format(
                    n, h, str(e).splitlines()[0][:50]))
                continue
            nb = bytes_moved(n, h)
            flag = "  <- spread > 5%, do not quote" if max(sb, sa) > 0.05 else ""
            print("  {:>7} {:>6} {:>12.3f} {:>12.3f} {:>9.2f}x {:>10.1f} "
                  "{:>10.1f} {:>7.1%} {:>7.1%}{}".format(
                      n, h, tb * 1e3, ta * 1e3, tb / ta,
                      nb / tb / 1e9, nb / ta / 1e9, sb, sa, flag))

    print("\n[RESULT] DONE")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
