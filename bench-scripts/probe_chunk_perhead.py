"""Per-head implied position: what index did each head actually rotate by?

WHERE THIS PICKS UP. Inverting the bad rotation showed cos^2+sin^2 = 0.9997, so
the arithmetic is right and the cos/sin are genuine table entries -- but
max|cos_h - cos_0| = 2.004, so the angle DIFFERS BETWEEN HEADS. That breaks the
override's central invariant: heads are tiled inside a token precisely so that
every head of a token shares one scalar position. Head 32's implied position was
30720 against a correct 6144, a difference of 24576, which is exactly
total_programs for this shape.

One head's number cannot distinguish a scalar read from the wrong place from a
vector that should never have been a vector. Thirty-two of them can: solve each
head's rotation separately and print the implied position for every one.

WHAT THE SHAPE OF THE ANSWER MEANS, fixed before the numbers are seen:

  all 32 equal, and wrong        -> a scalar read from the wrong address; the
                                    defect is one index expression
  an arithmetic progression      -> the position became a VECTOR over the head
                                    axis; the stride names the term that leaked
                                    the head index in
  equal to correct + row index   -> `rows` reached the position load, i.e. a
                                    per-head row offset was used where a
                                    per-token one belongs
  unstructured                   -> the load is reading past the end of
                                    position_ids and returning adjacent memory,
                                    which is a bounds bug, not an index bug

position_ids holds arange(num_tokens), so a value equal to its own index means
the read was in bounds; a value >= num_tokens can only have come from beyond the
end of the tensor. Both are reported, because they need different fixes.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_chunk_perhead.py
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
CS_ROWS = 8 * NUM_TOKENS


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
    total = NUM_TOKENS * tiles + NUM_TOKENS
    print("boundary program {}  token {}  heads {}..{}  H={}  tiles={}  total={}"
          .format(CAP, tok, head_base, head_base + H - 1, H, tiles, total))

    x = q0[tok, head_base:head_base + H].float().cpu()
    rsqrt = torch.rsqrt((x * x).sum(1) / HEAD_DIM + EPS)[:, None]
    pr = x[:, NOPE_DIM:].reshape(H, HALF, 2)
    e, o = pr[..., 0] * rsqrt, pr[..., 1] * rsqrt
    bad = q_bad[tok, head_base:head_base + H, NOPE_DIM:].float().cpu()
    ne, no = bad.reshape(H, HALF, 2)[..., 0], bad.reshape(H, HALF, 2)[..., 1]

    det = e * e + o * o
    cos_i = (e * ne + o * no) / det
    sin_i = (e * no - o * ne) / det
    cs_c = cs.float().cpu()

    print("\n  {:>4} {:>6} {:>11} {:>10} {:>12} {:>10}".format(
        "head", "abs", "implied pos", "err", "pos-correct", "in bounds"))
    print("  " + "-" * 60)
    implied = []
    for h in range(H):
        err = ((cs_c[:, :HALF] - cos_i[h][None, :]).abs().max(1).values
               + (cs_c[:, HALF:] - sin_i[h][None, :]).abs().max(1).values)
        p = int(err.argmin())
        implied.append(p)
        if h < 8 or h >= H - 4 or p - tok != implied[0] - tok:
            print("  {:>4} {:>6} {:>11} {:>10.2e} {:>12} {:>10}".format(
                h, head_base + h, p, float(err[p]), p - tok,
                "yes" if p < NUM_TOKENS else "NO"))
    ip = torch.tensor(implied)
    d = ip - tok
    print("\n  distinct implied positions: {}".format(int(torch.unique(ip).numel())))
    print("  delta from correct: min {}  max {}".format(int(d.min()), int(d.max())))
    diffs = (ip[1:] - ip[:-1])
    const = bool((diffs == diffs[0]).all()) if H > 1 else False
    print("  consecutive step constant: {}{}".format(
        const, "  step = {}".format(int(diffs[0])) if const else ""))
    oob = int((ip >= NUM_TOKENS).sum())
    print("  positions >= num_tokens (must have come from past the end): {} of {}"
          .format(oob, H))

    print()
    if int(torch.unique(ip).numel()) == 1:
        print("SCALAR, WRONG ADDRESS. Every head shares one position and it is")
        print("wrong by {}. One index expression is at fault.".format(int(d[0])))
        print("\n[RESULT] SCALAR_OFF_BY_{}".format(int(d[0])))
    elif const:
        print("THE POSITION BECAME A VECTOR over the head axis, step {}."
              .format(int(diffs[0])))
        print("Something carrying the head index reached the position load, which")
        print("the tiling exists specifically to keep scalar.")
        print("\n[RESULT] VECTOR_STEP_{}".format(int(diffs[0])))
    else:
        print("UNSTRUCTURED. The values are not one position and not a stride, so")
        print("the load is returning memory past the end of position_ids rather")
        print("than a mis-computed index. That is a bounds bug and the fix is not")
        print("an arithmetic correction.")
        print("\n[RESULT] UNSTRUCTURED_OOB")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
