"""Are the implied positions real, or nearest-neighbour artefacts? Control first.

WHY THIS EXISTS, AND WHAT IT DOUBTS. The per-head probe reported implied
positions 30720 and 36864 -- two distinct values across 32 heads -- and its own
verdict called that "unstructured". The verdict was wrong: two repeated values
are structure. But the reading underneath it is not yet safe to use, for a
reason that probe did not check.

  * the match errors were 1.1e-2 to 2.2e-2, not ~0
  * inverting bf16 output leaves ~2% noise of its own (cos^2+sin^2 ran 0.978 to
    1.025)
  * the search space was 65536 rows of 32 cosines

A nearest-neighbour search over a table that dense will return SOMETHING within
a small distance for any target at all, including pure noise. Until the null
distance is known, "best matching position p" is not evidence that position p
was ever read. Reading 30720 as a real index, and then editing the kernel around
it, would be building on an artefact.

THIS MEASURES THE NULL. Random unit (cos, sin) vectors -- angles the table never
contained -- are matched against the same table the same way. Their error
distribution is what "no real match" looks like here.

  observed error well below the null   -> the positions are real
  observed error inside the null       -> they are artefacts, and every number
                                          in the previous table is discarded

THE SECOND TEST, which does not depend on the first. Move the chunk boundary so
a DIFFERENT token goes bad, and invert again. A real index tracks the change in
some legible way; an artefact does not. Two boundaries that produce the same two
constants mean something; two that produce unrelated noise mean the inversion
has hit its precision floor and a different instrument is needed.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_chunk_control.py
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
HEAD_BYTES, EPS = 584, 1e-6
NUM_TOKENS, NUM_HEADS, BLOCK = 8192, 64, 64
CS_ROWS = 8 * NUM_TOKENS
CAPS = (12289, 8193, 20481)


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
    t = torch.arange(CS_ROWS, dtype=torch.float32, device=dev)
    fr = torch.einsum("i,j->ij", t, inv)
    cs = torch.cat((fr.cos(), fr.sin()), dim=-1)
    nb = (NUM_TOKENS + BLOCK - 1) // BLOCK + 1
    slot = torch.arange(NUM_TOKENS, dtype=torch.int64, device=dev)
    kc = torch.zeros(nb, BLOCK * HEAD_BYTES, dtype=torch.uint8, device=dev)
    cs_c = cs.float().cpu()

    def nearest(c, s):
        err = ((cs_c[:, :HALF] - c[None, :]).abs().max(1).values
               + (cs_c[:, HALF:] - s[None, :]).abs().max(1).values)
        i = int(err.argmin())
        return i, float(err[i])

    # ---------- 1. the null -------------------------------------------------
    print("=" * 74)
    print("1. NULL: what error does a target the table never held still get?")
    print("=" * 74)
    g = torch.Generator().manual_seed(7)
    nulls = []
    for _ in range(40):
        ang = torch.rand(HALF, generator=g) * 6.283185307
        nulls.append(nearest(ang.cos(), ang.sin())[1])
    nt = torch.tensor(nulls)
    print("  40 random angle vectors, matched against the same {} rows".format(CS_ROWS))
    print("  null match error:  min {:.3e}  median {:.3e}  max {:.3e}"
          .format(float(nt.min()), float(nt.median()), float(nt.max())))
    print("\n  observed errors in the per-head table were 1.1e-2 to 2.2e-2.")
    if float(nt.median()) <= 3e-2:
        print("  -> the null is the SAME SIZE. A match at that distance is not")
        print("     evidence of anything, and 30720 / 36864 must be discarded.")
        null_kills = True
    else:
        print("  -> the null is much larger, so the observed matches are real.")
        null_kills = False

    # ---------- 2. does the answer move with the boundary? ------------------
    print("\n" + "=" * 74)
    print("2. Move the boundary. A real index tracks it; an artefact does not.")
    print("=" * 74)

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
    H = mod.q_heads_per_program(NUM_HEADS)
    tiles = NUM_HEADS // H

    print("  {:>7} {:>7} {:>7} {:>28} {:>10}".format(
        "cap", "token", "heads", "distinct implied positions", "med err"))
    print("  " + "-" * 68)
    for cap in CAPS:
        q_bad = run(cap)
        tok, hb = cap // tiles, (cap % tiles) * H
        d = q_ref != q_bad
        toks = sorted(set(d.nonzero()[:, 0].tolist()))
        if tok not in toks:
            print("  {:>7} {:>7} {:>7} {:>28} {:>10}".format(
                cap, tok, "-", "predicted token not among {}".format(toks[:3]), "-"))
            continue
        x = q0[tok, hb:hb + H].float().cpu()
        rs = torch.rsqrt((x * x).sum(1) / HEAD_DIM + EPS)[:, None]
        pr = x[:, NOPE_DIM:].reshape(H, HALF, 2)
        e, o = pr[..., 0] * rs, pr[..., 1] * rs
        bad = q_bad[tok, hb:hb + H, NOPE_DIM:].float().cpu()
        ne, no = bad.reshape(H, HALF, 2)[..., 0], bad.reshape(H, HALF, 2)[..., 1]
        det = e * e + o * o
        ci, si = (e * ne + o * no) / det, (e * no - o * ne) / det
        ps, es = [], []
        for h in range(H):
            p, er = nearest(ci[h], si[h])
            ps.append(p)
            es.append(er)
        u = sorted(set(ps))
        print("  {:>7} {:>7} {:>7} {:>28} {:>10.2e}".format(
            cap, tok, "{}..{}".format(hb, hb + H - 1),
            str(u[:4]) + ("..." if len(u) > 4 else ""),
            float(torch.tensor(es).median())))
        del q_bad

    print()
    if null_kills:
        print("VERDICT: the implied positions are not established. The inversion")
        print("reached its precision floor -- bf16 output leaves ~2% in the solved")
        print("cos/sin, and this table is dense enough to absorb that. What still")
        print("stands from the earlier runs, because it did not depend on the")
        print("search, is that the bad program APPLIES A ROTATION (norm 0.9997)")
        print("with an angle that DIFFERS ACROSS HEADS. Chase that with an")
        print("instrument that does not need a nearest-neighbour match.")
        print("\n[RESULT] POSITIONS_NOT_ESTABLISHED")
    else:
        print("VERDICT: matches are well inside the null, so the positions are")
        print("real and the table above says how they move with the boundary.")
        print("\n[RESULT] POSITIONS_REAL")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
