"""The defect is Q-arm specific. Close the last gap to a minimal reproducer.

WHAT NARROWED IT. The launch configuration is not the variable -- every
num_warps is broken under chunking and every single launch agrees, so that is
crossed off along with pid arithmetic, a scalar gather, a two-level table
lookup, a head-axis broadcast, and a masked store followed by a reload of a
disjoint column range, all cleared by the ladder.

THE CLUE THAT WAS SITTING IN THE DATA. With cap 8193 the third launch begins at
program 16386, which is past q_programs and therefore a KV program -- and
k_cache came back byte-identical every time. A launch whose first program lands
on the KV arm is fine. Only the Q arm is corrupted. So whatever it is, the KV
arm does not have it.

WHAT THE Q ARM HAS THAT THE KV ARM AND THE CLEAN LADDER RUNG DO NOT:
bfloat16 storage, and tl.split/tl.join over a three-dimensional pair offset.
Ladder rung 5 had the store-and-reload structure but in float32 and with
neither split nor join, which is why clearing it settled less than it looked.

TWO TESTS, EITHER OF WHICH IS WORTH THE RUN.

A. Launch only the Q programs. The host loop is replicated here, so dropping
   the KV programs from the grid needs no source change and no program ever
   takes the else arm. Clean means the presence of the other arm matters --
   which would be surprising given the asymmetry above, and worth knowing
   before anything is rewritten. Broken means the Q arm fails on its own and
   the branch is irrelevant.

B. A self-contained kernel that is the Q arm and nothing else: bf16 q, RMS
   scale, masked NoPE store, three-dimensional pair offset, tl.split, rotate,
   tl.join. If this breaks under chunking it is a minimal reproducer -- small
   enough to bisect inside, and small enough to hand to the vendor. If it stays
   clean, the real Q arm has something this copy does not, and the copy is the
   thing to grow.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_chunk_qarm.py
"""

import importlib
import os
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

HEAD_DIM, ROPE_DIM, HEAD_BYTES, EPS = 512, 64, 584, 1e-6
NUM_TOKENS, NUM_HEADS, BLOCK = 8192, 64, 64
REPEATS = 3


@triton.jit
def q_arm_only(q, src, table, pid_offset, tiles,
               V: tl.constexpr, HH: tl.constexpr, W: tl.constexpr):
    """The Q arm, transcribed and nothing else. bf16 q, split/join, masked store."""
    pid = tl.program_id(0).to(tl.int64) + pid_offset
    tok = pid // tiles
    rows = tok * (tiles * HH) + (pid % tiles) * HH + tl.arange(0, HH)
    col = tl.arange(0, W)
    blk = tl.load(q + rows[:, None] * W + col[None, :]).to(tl.float32)
    var = tl.sum(blk * blk, axis=1) / W
    rs = tl.rsqrt(var + 1e-6)
    blk = blk * rs[:, None]
    tl.store(q + rows[:, None] * W + col[None, :],
             blk.to(tl.bfloat16), mask=col[None, :] < W - V)
    p = tl.load(src + tok)
    half = tl.arange(0, V // 2)
    c = tl.load(table + p * V + half)
    s = tl.load(table + p * V + V // 2 + half)
    pair_off = (rows[:, None, None] * W + (W - V)
                + half[None, :, None] * 2 + tl.arange(0, 2)[None, None, :])
    pair = tl.load(q + pair_off).to(tl.float32)
    e, o = tl.split(pair)
    e = e * rs[:, None]
    o = o * rs[:, None]
    ne = e * c[None, :] - o * s[None, :]
    no = e * s[None, :] + o * c[None, :]
    tl.store(q + pair_off, tl.join(ne, no).to(tl.bfloat16))


def main():
    import flaggems_vllm

    mod = importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.fused"
        ".fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    )
    kern = mod.fused_qnorm_rope_kv_insert_kernel
    dev = flaggems_vllm.device
    sync = flaggems_vllm.runtime.torch_device_fn.synchronize

    torch.manual_seed(0)
    q0 = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    kv = torch.randn(NUM_TOKENS, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    pos = torch.arange(NUM_TOKENS, dtype=torch.int64, device=dev)
    inv = 1.0 / (10000.0 ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32,
                                          device=dev) / ROPE_DIM))
    t = torch.arange(NUM_TOKENS, dtype=torch.float32, device=dev)
    fr = torch.einsum("i,j->ij", t, inv)
    cs = torch.cat((fr.cos(), fr.sin()), dim=-1)
    nb = (NUM_TOKENS + BLOCK - 1) // BLOCK + 1
    slot = torch.arange(NUM_TOKENS, dtype=torch.int64, device=dev)
    kc0 = torch.zeros(nb, BLOCK * HEAD_BYTES, dtype=torch.uint8, device=dev)

    H = mod.q_heads_per_program(NUM_HEADS)
    tiles = NUM_HEADS // H
    q_programs = NUM_TOKENS * tiles
    CAP = q_programs // 2 + 1
    print("H={} tiles={} q_programs={}  cap={} -> first program of launch 2 is"
          " token {}, tile {}".format(H, tiles, q_programs, CAP,
                                      CAP // tiles, CAP % tiles))

    def check(label, runner):
        ref = runner(1 << 30)
        runs = [runner(CAP) for _ in range(REPEATS)]
        vs_ref = int((ref != runs[0]).sum())
        vs_self = max(int((runs[0] != r).sum()) for r in runs[1:])
        clean = vs_ref == 0 and vs_self == 0
        print("  {:<34} {:>10} {:>14}   {}".format(
            label, vs_ref, "max {}".format(vs_self), "OK" if clean else "BROKEN"))
        if not clean:
            idx = (ref != runs[0]).nonzero()
            print("      tokens {}  dims {}..{}".format(
                sorted(set(idx[:, 0].tolist()))[:4],
                int(idx[:, -1].min()), int(idx[:, -1].max())))
        del ref, runs
        flaggems_vllm.runtime.torch_device_fn.empty_cache()
        return clean

    print("\n  {:<34} {:>10} {:>14}   verdict".format(
        "test", "vs single", "3 vs 3"))
    print("  " + "-" * 74)

    # ---- A. real kernel, Q programs only (never enters the else arm) -------
    def real_q_only(cap):
        q, kc = q0.clone(), kc0.clone()
        kcb = kc.view(torch.bfloat16)
        for off in range(0, q_programs, cap):
            grid = min(cap, q_programs - off)
            kern[(grid,)](q, kv, kc, kcb, slot, pos, cs, EPS, BLOCK,
                          NUM_HEADS, kc.stride(0), off, q_programs, tiles, H,
                          num_warps=1, num_stages=1)
        sync()
        return q

    a_clean = check("A. real kernel, Q programs only", real_q_only)

    # ---- B. a standalone transcription of the Q arm ------------------------
    def standalone(cap):
        q = q0.clone()
        for off in range(0, q_programs, cap):
            grid = min(cap, q_programs - off)
            q_arm_only[(grid,)](q, pos, cs, off, tiles,
                                ROPE_DIM, H, HEAD_DIM,
                                num_warps=1, num_stages=1)
        sync()
        return q

    try:
        b_clean = check("B. standalone Q-arm transcription", standalone)
    except Exception as e:
        print("  {:<34} {:>10}   failed: {}".format(
            "B. standalone Q-arm transcription", "-",
            str(e).splitlines()[0][:44]))
        b_clean = None

    print()
    if not a_clean and b_clean is False:
        print("MINIMAL REPRODUCER FOUND. The Q arm alone, with no KV programs in")
        print("the grid and no other arm in the kernel, corrupts its first")
        print("program per launch. Bisect inside this standalone kernel -- drop")
        print("split/join, then bf16, then the masked store -- and the survivor")
        print("is the defect. It is also small enough to report as-is.")
        print("\n[RESULT] MINIMAL_REPRO")
    elif not a_clean and b_clean:
        print("The real Q arm is broken but this transcription is not, so the")
        print("copy is missing whatever matters. Diff them and grow the copy")
        print("rather than trusting it.")
        print("\n[RESULT] TRANSCRIPTION_TOO_SMALL")
    elif a_clean:
        print("Q PROGRAMS ALONE ARE CLEAN. Dropping the KV programs from the")
        print("grid fixes it, so the defect needs BOTH arms present -- despite")
        print("k_cache always being correct. The shared grid is the variable,")
        print("and splitting Q and KV into separate launches is a candidate fix.")
        print("\n[RESULT] NEEDS_BOTH_ARMS")
    else:
        print("Inconclusive -- read the rows above.")
        print("\n[RESULT] INCONCLUSIVE")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
