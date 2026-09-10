"""17 tokens, 64 heads: the smallest non-deterministic case. Characterise it.

WHY HERE, AND NOT AT THE BIG SHAPES. Running the operator twice on identical
input over the whole grid found it self-consistent at 1024, 8192, 32768 and
65536 tokens -- three runs each, zero differences -- and non-deterministic at
exactly one place: 17 tokens at 64 heads, both block sizes, token 0, all 512
dims. 17 tokens at 128 heads is clean.

This case predates the 2-D grid change: `test_backend_override_matches_
reference[64-64-17]` was already failing, and flipping between runs, before any
edit. It was written off as flakiness and it was the defect all along, at a
shape that runs in seconds -- while several rounds went into 131072-token
shapes that take minutes and make probes run out of memory.

64 heads gives tiles_per_token = 2 with heads_per_program = 32; 128 heads gives
4 and is clean. So the count of programs per token is a live variable, and 17 is
odd, which makes the Q region 34 programs and the total 51.

WHAT IS MEASURED. Twenty runs on one input, each compared against the first,
reporting for every differing element how often it differs and which heads and
dims are involved -- so a boundary effect, a whole-head effect and scattered
corruption are told apart rather than guessed at. Then the same input at
neighbouring token counts, 16 through 20, to see whether 17 is special or
whether every count behaves this way and only this one was sampled.

Memory-lean on purpose: one reference plus one run live at a time. The previous
probe kept three copies of q and ran out of memory at the large shapes, which
cost the measurement there.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_17x64.py
"""

import os
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

HEAD_DIM, ROPE_DIM, HEAD_BYTES, EPS = 512, 64, 584, 1e-6
RUNS = 20


def build(n, h, b, dev):
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
    sync = flaggems_vllm.runtime.torch_device_fn.synchronize
    mod_ok = True
    try:
        from flaggems_vllm.runtime.backend._ascend.fused import (  # noqa: F401
            fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert as _chk,
        )
    except Exception:
        mod_ok = False
    print("override importable: {}".format(mod_ok))

    def once(inp):
        q, kv, kc, slot, pos, cs, b = inp
        qq, kk = q.clone(), kc.clone()
        impl(qq, kv, kk, slot, pos, cs, EPS, b)
        sync()
        return qq, kk

    # ---- 1. twenty runs at 17 x 64 ---------------------------------------
    print("\n" + "=" * 76)
    print("1. Twenty runs, 17 tokens x 64 heads, block 64")
    print("=" * 76)
    inp = build(17, 64, 64, dev)
    ref_q, ref_kc = once(inp)
    freq = torch.zeros_like(ref_q, dtype=torch.int32)
    diff_runs = 0
    for _ in range(RUNS - 1):
        gq, gkc = once(inp)
        m = gq != ref_q
        if int(m.sum()):
            diff_runs += 1
        freq += m.to(torch.int32)
        del gq, gkc
    print("  runs differing from the first: {} of {}".format(diff_runs, RUNS - 1))
    ever = freq > 0
    if int(ever.sum()):
        toks = ever.any(2).any(1).nonzero().flatten()
        heads = ever.any(2).any(0).nonzero().flatten()
        dims = ever.any(0).any(0).nonzero().flatten()
        print("  tokens involved : {}".format(toks.tolist()))
        print("  heads involved  : {} of 64  ({}..{})".format(
            int(heads.numel()), int(heads.min()), int(heads.max())))
        print("  dims involved   : {} of 512 ({}..{})".format(
            int(dims.numel()), int(dims.min()), int(dims.max())))
        print("  elements ever differing: {}".format(int(ever.sum())))
        print("  worst element differs in {} of {} runs".format(
            int(freq.max()), RUNS - 1))
        nope = int(ever[:, :, :448].sum())
        rope = int(ever[:, :, 448:].sum())
        print("  split: {} in NoPE (0..447), {} in RoPE (448..511)".format(nope, rope))
        h_all = int((ever.any(2).sum(1) == 64).sum())
        print("  tokens where EVERY head is involved: {}".format(h_all))
    else:
        print("  no differences in 20 runs -- it did not reproduce this time,")
        print("  which for a flaky defect is data, not absence. Re-run before")
        print("  concluding anything.")
    del inp, ref_q, ref_kc, freq
    flaggems_vllm.runtime.torch_device_fn.empty_cache()

    # ---- 2. neighbouring token counts ------------------------------------
    print("\n" + "=" * 76)
    print("2. Is 17 special, or does every small odd count do this?")
    print("=" * 76)
    print("  {:>7} {:>6} {:>7} {:>8} {:>12} {:>14}".format(
        "tokens", "heads", "q_prog", "total", "runs differ", "elements"))
    print("  " + "-" * 62)
    for n in (15, 16, 17, 18, 19, 20, 33, 64, 65):
        for h in (64, 128):
            inp = build(n, h, 64, dev)
            tiles = h // 32
            try:
                ref = once(inp)[0]
                bad, tot = 0, 0
                for _ in range(5):
                    g = once(inp)[0]
                    d = int((g != ref).sum())
                    bad += 1 if d else 0
                    tot = max(tot, d)
                    del g
                print("  {:>7} {:>6} {:>7} {:>8} {:>12} {:>14}".format(
                    n, h, n * tiles, n * tiles + n, "{}/5".format(bad), tot))
                del ref
            except Exception as e:
                print("  {:>7} {:>6}   {}".format(n, h, str(e).splitlines()[0][:44]))
            del inp
            flaggems_vllm.runtime.torch_device_fn.empty_cache()

    print("\n  A defect at 17 and not at 16 or 18 points at the exact program")
    print("  count; one that appears at every small count points at something")
    print("  about small grids in general. The table above decides which, and")
    print("  a clean 128-head column throughout keeps tiles_per_token in frame.")
    print("\n[RESULT] CHARACTERISED")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
