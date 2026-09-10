"""Invert the bad rotation instead of guessing what it was.

WHY THIS EXISTS. The candidate-matching probe checked ten specific failures --
wrong position, missing scale, double rotation, swapped cos/sin, raw data -- and
NONE reproduced the bad bytes, at relative differences of 450 to 980. Its host
model is sound: it reproduced the GOOD run exactly. So the model is a valid
yardstick and the candidate set was simply too small. Guessing more candidates
is the same move again; this inverts the observation instead.

THE INVERSION. GPT-J RoPE on a pair is a 2x2 rotation with known inputs:

    new_e = e*cos - o*sin
    new_o = e*sin + o*cos

e and o are known (the original q times the RMS scale, both of which the good
run confirms). new_e and new_o are what the bad run actually wrote. Two
equations, two unknowns, and the system is non-singular whenever e^2+o^2 > 0:

    cos = (e*new_e + o*new_o) / (e^2 + o^2)
    sin = (e*new_o - o*new_e) / (e^2 + o^2)

So the (cos, sin) the kernel effectively applied can be READ OFF, per pair, with
no hypothesis at all.

WHAT THE ANSWER MEANS, decided before the numbers are seen:

  * cos^2 + sin^2 ~= 1 and the same across all 32 heads
        -> it IS a rotation, by a wrong angle. A position was read wrongly, or
           cos/sin were fetched from the wrong place. Then the implied angle is
           searched for in the table, INCLUDING positions past the end of it,
           because an out-of-range position reading whatever follows the cache
           is exactly the kind of thing that reproduces nothing.
  * cos^2 + sin^2 far from 1, or inconsistent across heads
        -> it is NOT a rotation. No amount of position hunting will explain it,
           and the defect is in the arithmetic, the data the pairs were loaded
           from, or ordering against the NoPE store.

THE SECOND HALF. The table is rebuilt with four times the rows so that a
position past num_tokens still lands in real, known memory. If the implied
angle then matches some position p, p itself names the arithmetic slip.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_chunk_invert.py
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
CAP = 12289
CS_ROWS = 4 * NUM_TOKENS       # deliberately long: out-of-range stays findable


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

    def run(cap):
        qq, kk = q0.clone(), kc.clone()
        mod.MAX_PROGRAMS_PER_LAUNCH = cap
        try:
            impl(qq, kv, kk, slot, pos, cs, EPS, BLOCK)
            sync()
        finally:
            mod.MAX_PROGRAMS_PER_LAUNCH = real_cap
        return qq

    q_ref, q_bad = run(1 << 30), run(CAP)

    H = mod.q_heads_per_program(NUM_HEADS)
    tiles = NUM_HEADS // H
    tok, head_base = CAP // tiles, (CAP % tiles) * H
    print("boundary program {}  ->  token {}, heads {}..{}"
          .format(CAP, tok, head_base, head_base + H - 1))

    d = q_ref != q_bad
    if int(d.sum()) == 0:
        print("no difference -- nothing to invert (did the cap take effect?)")
        print("\n[RESULT] NO_DIFFERENCE")
        return
    idx = d.nonzero()
    print("differing: {} elements, tokens {}, heads {}..{}, dims {}..{}\n"
          .format(int(d.sum()), sorted(set(idx[:, 0].tolist()))[:4],
                  int(idx[:, 1].min()), int(idx[:, 1].max()),
                  int(idx[:, 2].min()), int(idx[:, 2].max())))

    x = q0[tok, head_base:head_base + H].float().cpu()
    rsqrt = torch.rsqrt((x * x).sum(1) / HEAD_DIM + EPS)[:, None]
    pr = x[:, NOPE_DIM:].reshape(H, HALF, 2)
    e, o = pr[..., 0] * rsqrt, pr[..., 1] * rsqrt          # [H, HALF]

    bad = q_bad[tok, head_base:head_base + H, NOPE_DIM:].float().cpu()
    new_e, new_o = bad.reshape(H, HALF, 2)[..., 0], bad.reshape(H, HALF, 2)[..., 1]

    det = e * e + o * o
    good = det > 1e-12
    cos_i = torch.where(good, (e * new_e + o * new_o) / det, torch.zeros_like(det))
    sin_i = torch.where(good, (e * new_o - o * new_e) / det, torch.zeros_like(det))
    norm = cos_i * cos_i + sin_i * sin_i

    n_ok = int(good.sum())
    print("  invertible pairs: {} of {}".format(n_ok, good.numel()))
    print("  cos^2+sin^2  median {:.4f}   min {:.4f}   max {:.4f}"
          .format(float(norm[good].median()), float(norm[good].min()),
                  float(norm[good].max())))

    is_rot = bool((norm[good] - 1).abs().median() < 0.05)
    # consistent across heads? compare each head's implied angle to head 0's
    spread = float((cos_i[good.all(1)] - cos_i[good.all(1)][:1]).abs().max()) \
        if int(good.all(1).sum()) > 1 else float("nan")
    print("  same angle across heads: max |cos_h - cos_0| = {:.4e}".format(spread))

    if not is_rot:
        print("\n  cos^2+sin^2 is NOT 1 -- the bad program did not apply a rotation.")
        print("  A wrong position cannot explain this; stop hunting positions.")
        print("  Look at what the pairs were loaded from, and at the ordering")
        print("  against the NoPE store that precedes this load.")
        print("\n[RESULT] NOT_A_ROTATION")
        return

    print("\n  It IS a rotation by a wrong angle. Searching the table, including")
    print("  positions past num_tokens ({} rows built).".format(CS_ROWS))
    cs_c = cs.float().cpu()
    tgt_c = cos_i[good.all(1)][0] if int(good.all(1).sum()) else cos_i[0]
    tgt_s = sin_i[good.all(1)][0] if int(good.all(1).sum()) else sin_i[0]
    err = ((cs_c[:, :HALF] - tgt_c[None, :]).abs().max(1).values
           + (cs_c[:, HALF:] - tgt_s[None, :]).abs().max(1).values)
    p = int(err.argmin())
    print("  best matching position p = {}   (max abs error {:.3e})"
          .format(p, float(err[p])))
    print("  correct position would be {}   difference {}".format(tok, p - tok))
    for label, val in (("boundary program", CAP), ("tiles_per_token", tiles),
                       ("head_base", head_base), ("token*tiles", tok * tiles)):
        if p == val:
            print("  ** p equals {} **".format(label))
    print("\n[RESULT] ROTATION_WRONG_POSITION_{}".format(p))


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
