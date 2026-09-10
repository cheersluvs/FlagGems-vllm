"""Does the launch configuration decide whether chunking corrupts Q?

WHY THIS, AND WHY NOT ANOTHER LADDER RUNG. Building up from a trivial kernel
cleared five ingredients -- pid_offset arithmetic, a scalar gather off pid, a
two-level table lookup, a broadcast across the head axis, and a masked store
followed by a reload of a disjoint column range. All five were identical single
versus chunked and repeatable three times over. So the cause lives in the gap
that is left, and the gap is now smaller than the climb: it is faster to ablate
DOWN from the real kernel than to keep adding rungs.

Of what remains -- num_warps=1, tl.split/tl.join, the fp8 encoder, the KV arm
sharing the grid, bf16 storage -- the launch configuration is tested first for
one reason: on this backend a wrong answer that depends on num_warps is the
recorded signature of an intra-program store-then-load hazard. That is a
specific, previously observed failure mode, not a fresh guess, and one line
decides it.

HOW. The real kernel is called directly, with the host loop replicated here
rather than edited in place, so nothing about the shipped source changes. For
each launch configuration: run the work as ONE launch, then chunked three
times, and compare chunked against single and the chunked runs against each
other.

The single-launch results are also compared ACROSS configurations. A correct
kernel cannot depend on num_warps, so if those disagree the defect is not about
chunking at all and every per-boundary conclusion so far needs revisiting --
which is worth catching here rather than after a fix is written.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_chunk_warps.py
"""

import importlib
import os
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

HEAD_DIM, ROPE_DIM, HEAD_BYTES, EPS = 512, 64, 584, 1e-6
NUM_TOKENS, NUM_HEADS, BLOCK = 8192, 64, 64
CONFIGS = ((1, 1), (2, 1), (4, 1), (1, 2), (4, 2))
REPEATS = 3


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
    total = q_programs + NUM_TOKENS
    CAP = q_programs // 2 + 1        # one boundary, inside a token, inside Q
    print("H={} tiles={} q_programs={} total={}  cap={} -> boundary token {}"
          .format(H, tiles, q_programs, total, CAP, CAP // tiles))

    def run(cap, warps, stages):
        q, kc = q0.clone(), kc0.clone()
        kcb = kc.view(torch.bfloat16)
        for off in range(0, total, cap):
            grid = min(cap, total - off)
            kern[(grid,)](q, kv, kc, kcb, slot, pos, cs, EPS, BLOCK,
                          NUM_HEADS, kc.stride(0), off, q_programs, tiles, H,
                          num_warps=warps, num_stages=stages)
        sync()
        return q, kc

    print("\n  {:>6} {:>7} {:>18} {:>24} {:>9}".format(
        "warps", "stages", "chunked vs single", "3 chunked vs each other", "verdict"))
    print("  " + "-" * 70)
    singles = {}
    any_clean, all_broken = False, True
    for warps, stages in CONFIGS:
        try:
            ref_q, ref_kc = run(1 << 30, warps, stages)
            singles[(warps, stages)] = ref_q
            runs = [run(CAP, warps, stages) for _ in range(REPEATS)]
            vs_ref = int((ref_q != runs[0][0]).sum())
            vs_self = max(int((runs[0][0] != r[0]).sum()) for r in runs[1:])
            kc_bad = max(int((ref_kc != r[1]).sum()) for r in runs)
            clean = vs_ref == 0 and vs_self == 0 and kc_bad == 0
            any_clean = any_clean or clean
            all_broken = all_broken and not clean
            print("  {:>6} {:>7} {:>18} {:>24} {:>9}".format(
                warps, stages, vs_ref, "max {}".format(vs_self),
                "OK" if clean else "BROKEN"))
            del runs, ref_kc
            flaggems_vllm.runtime.torch_device_fn.empty_cache()
        except Exception as e:
            print("  {:>6} {:>7}   failed: {}".format(
                warps, stages, str(e).splitlines()[0][:52]))

    # A correct kernel cannot depend on the launch configuration.
    print("\n  single-launch results across configurations:")
    keys = sorted(singles)
    base = singles[keys[0]]
    disagree = [(k, int((base != singles[k]).sum())) for k in keys[1:]]
    for k, n in disagree:
        print("    {} vs {}: {} differing".format(keys[0], k, n))
    single_stable = all(n == 0 for _, n in disagree)
    print("    -> {}".format("all agree" if single_stable
                             else "THEY DISAGREE"))

    print()
    if not single_stable:
        print("THE SINGLE LAUNCH ITSELF DEPENDS ON num_warps. A correct kernel")
        print("cannot, so chunking is not the whole defect and the boundary")
        print("story needs revisiting before any fix is written.")
        print("\n[RESULT] SINGLE_LAUNCH_WARP_DEPENDENT")
    elif any_clean:
        print("SOME LAUNCH CONFIGURATION IS CLEAN under chunking. That confirms")
        print("the hazard is launch-configuration dependent -- the recorded")
        print("signature of an intra-program store-then-load on this backend --")
        print("and hands a candidate workaround whose cost can be measured.")
        print("\n[RESULT] WARP_DEPENDENT")
    else:
        print("EVERY CONFIGURATION IS BROKEN under chunking, and every single")
        print("launch agrees. So the launch configuration is NOT the variable;")
        print("cross it off and ablate the next difference -- tl.split/tl.join,")
        print("the fp8 encoder, or the KV arm sharing the grid.")
        print("\n[RESULT] NOT_WARP_DEPENDENT")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
