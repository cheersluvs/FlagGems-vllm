"""Which wrong thing does the first program of a chunk do to RoPE?

WHAT IS ALREADY MEASURED, and is not re-argued here. Splitting the same work
across launches makes the FIRST PROGRAM of every launch after the first compute
its RoPE region wrong: 2048 elements per bad program (32 heads x 64 RoPE dims),
the NoPE region correct, k_cache byte-identical. Predicted and observed bad
tokens matched on 8 of 8 configurations -- count, first and last.

WHAT THIS DECIDES. Whether the bad program reads the wrong inputs (position,
cos, sin) or does the wrong arithmetic on the right ones.

HOW, AND WHY THIS SHAPE OF PROBE. It does not instrument the kernel: adding a
store to dump intermediates changes the program, and on this backend changing
the program changes what compiles -- an in-loop `tl.where`, an early return and
a device function each break it, so an instrumented kernel is not evidence about
the real one. Instead every candidate failure is COMPUTED ON THE HOST from the
original q and compared against the bytes the bad run actually produced. A
hypothesis either reproduces them exactly or it is out. That is a decision, not
a story.

THE CANDIDATES. Each is a specific thing that could go wrong at a chunk's first
program, and they give numerically very different answers:

  correct        rotate(normalised pairs, position = token_idx)
  pos_*          same, but the position read is some other index -- the
                 chunk-local program id, the offset, zero, a neighbour. This is
                 what a pid arithmetic slip looks like.
  no_rsqrt       rotated, but the RMS scale never applied to the pairs
  unrotated      scaled, never rotated
  double_rotate  rotated twice
  cos_sin_swap   cos and sin exchanged
  raw            q untouched

A match on `pos_*` means the address arithmetic is wrong and the fix is in how
pid becomes token_idx. A match on `no_rsqrt`, `double_rotate` or `unrotated`
means the arithmetic or the ordering is wrong and pid is fine. No match at all
means the failure is not a clean substitution -- worth knowing before anyone
edits the kernel on a hunch.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_chunk_rope.py
"""

import importlib
import os
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

HEAD_DIM, NOPE_DIM, ROPE_DIM, HALF = 512, 448, 64, 32
HEAD_BYTES = 584
EPS = 1e-6

NUM_TOKENS, NUM_HEADS, BLOCK = 8192, 64, 64
CAP = 12289          # one boundary inside Q; the bad token is 12289 // 2 = 6144


def rope(even, odd, cos, sin):
    return (even * cos - odd * sin, even * sin + odd * cos)


def main():
    import flaggems_vllm

    mod = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.fused"
        ".fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    )
    impl = mod.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    dev = flaggems_vllm.device
    sync = flaggems_vllm.runtime.torch_device_fn.synchronize
    real_cap = mod.MAX_PROGRAMS_PER_LAUNCH

    torch.manual_seed(0)
    q0 = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(NUM_TOKENS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    pos = torch.arange(NUM_TOKENS, dtype=torch.int64, device=dev)
    inv = 1.0 / (10000.0 ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32,
                                          device=dev) / ROPE_DIM))
    t = torch.arange(max(4096, NUM_TOKENS), dtype=torch.float32, device=dev)
    fr = torch.einsum("i,j->ij", t, inv)
    cs = torch.cat((fr.cos(), fr.sin()), dim=-1)
    nb = (NUM_TOKENS + BLOCK - 1) // BLOCK + 1
    slot = torch.arange(NUM_TOKENS, dtype=torch.int64, device=dev)
    kc = torch.zeros(nb, BLOCK * HEAD_BYTES, dtype=torch.uint8, device=dev)

    def run(cap):
        qq, kk = q0.clone(), kc.clone()
        mod.MAX_PROGRAMS_PER_LAUNCH = cap
        try:
            impl(qq, kv, kk, slot, pos, cs, EPS, BLOCK)
            sync()
        finally:
            mod.MAX_PROGRAMS_PER_LAUNCH = real_cap
        return qq

    q_ref = run(1 << 30)
    q_bad = run(CAP)

    tiles = NUM_HEADS // mod.q_heads_per_program(NUM_HEADS)
    H = mod.q_heads_per_program(NUM_HEADS)
    tok = CAP // tiles
    head_base = (CAP % tiles) * H
    print("chunk boundary at program {}  ->  token {}, heads {}..{}"
          .format(CAP, tok, head_base, head_base + H - 1))

    d = (q_ref != q_bad)
    idx = d.nonzero()
    print("differing elements {}   tokens {}   heads {}..{}   dims {}..{}"
          .format(int(d.sum()),
                  sorted(set(idx[:, 0].tolist())),
                  int(idx[:, 1].min()), int(idx[:, 1].max()),
                  int(idx[:, 2].min()), int(idx[:, 2].max())))

    # ---- rebuild every candidate on the host, in float32, from the ORIGINAL q
    x = q0[tok, head_base:head_base + H].float().cpu()          # [H, 512]
    cs_c = cs.float().cpu()
    got = q_bad[tok, head_base:head_base + H, NOPE_DIM:].float().cpu()
    ref = q_ref[tok, head_base:head_base + H, NOPE_DIM:].float().cpu()

    rsqrt = torch.rsqrt((x * x).sum(1) / HEAD_DIM + EPS)[:, None]   # [H,1]
    raw_pairs = x[:, NOPE_DIM:].reshape(H, HALF, 2)
    even_raw, odd_raw = raw_pairs[..., 0], raw_pairs[..., 1]
    even_n, odd_n = even_raw * rsqrt, odd_raw * rsqrt

    def pack(e, o):
        return torch.stack((e, o), dim=-1).reshape(H, ROPE_DIM).to(torch.bfloat16).float()

    cands = {}
    for p in sorted({tok, 0, 1, tok - 1, tok + 1, CAP, CAP % tiles,
                     CAP - tok, tok * tiles, (CAP // tiles) * tiles}):
        if 0 <= p < cs_c.shape[0]:
            c, s = cs_c[p, :HALF], cs_c[p, HALF:]
            name = "correct" if p == tok else "pos={}".format(p)
            cands[name] = pack(*rope(even_n, odd_n, c, s))
    c, s = cs_c[tok, :HALF], cs_c[tok, HALF:]
    cands["no_rsqrt"] = pack(*rope(even_raw, odd_raw, c, s))
    cands["unrotated"] = pack(even_n, odd_n)
    e1, o1 = rope(even_n, odd_n, c, s)
    cands["double_rotate"] = pack(*rope(e1, o1, c, s))
    cands["cos_sin_swap"] = pack(*rope(even_n, odd_n, s, c))
    cands["raw"] = pack(even_raw, odd_raw)

    print("\n  {:<18} {:>12} {:>16} {:>12}".format(
        "candidate", "exact match", "max rel diff", "vs BAD?"))
    print("  " + "-" * 62)
    winner = None
    for name in sorted(cands, key=lambda k: (k != "correct", k)):
        v = cands[name]
        eq = bool(torch.equal(v, got))
        rel = float(((v - got).abs() / got.abs().clamp(min=1e-30)).max())
        if eq and name != "correct":
            winner = name
        print("  {:<18} {:>12} {:>16.4e} {:>12}".format(
            name, str(eq), rel, "MATCH" if eq else ""))

    # sanity: the reference run must equal the `correct` candidate
    ok_ref = torch.equal(cands["correct"], ref)
    print("\n  reference run reproduced by `correct`: {}".format(ok_ref))
    if not ok_ref:
        print("  The host model does not reproduce the GOOD run either, so it is")
        print("  not a valid yardstick -- fix the model before reading the table.")
        print("\n[RESULT] MODEL_INVALID")
        return

    print()
    if winner:
        print("THE BAD PROGRAM IS DOING: {}".format(winner))
        if winner.startswith("pos="):
            print("It reads the wrong position -- the defect is in how pid becomes")
            print("token_idx for the first program of a chunk, not in the rotation.")
        else:
            print("It reads the right position and does the wrong arithmetic.")
        print("\n[RESULT] IDENTIFIED_{}".format(winner.replace("=", "_")))
    else:
        print("NO CANDIDATE REPRODUCES THE BAD OUTPUT EXACTLY.")
        print("The failure is not a clean substitution of one input or one step.")
        print("Do not edit the kernel on a hunch -- widen the candidate set first.")
        print("\n[RESULT] NO_CANDIDATE")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
