"""Failure rates for the shipped override and for a 20-line standalone kernel.

WHY THE OWNERSHIP TEST CHANGED. Comparing against the generic Triton kernel is
not available: it does not compile on this backend at all --

    ops/fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py:107:
    error: Casting pointers with unmatched bitwidth!

which is why the override exists. The previous probe also threw away the
override's numbers whenever the generic raised, because both were computed in
one try block. Each implementation is measured independently here; a failure in
one is reported and the other still counts.

WHAT WAS DISMISSED TOO EARLY. A standalone transcription of the Q arm was
unstable as a single launch, and that was read as "a neighbouring bug" on the
assumption that the real defect only appeared when launches were split. That
assumption is now known to be wrong: the defect is a shape-dependent
probability that lands on program 0, and the evidence for "single launch is
clean" was five runs at one shape, 8192 x 64, which is simply a shape with a
low rate. The standalone kernel's instability fits the same defect, so it is
measured here beside the real one rather than discarded.

If both show it, that 20-line kernel is the reproducer to send to the vendor:
no fp8, no paged cache, no KV arm, no vLLM, nothing that needs this repository.

WHAT THESE NUMBERS ARE, AND ARE NOT. Counts out of thirty runs on one input,
per shape. A zero here is weak evidence: with rates of 1/5 and 2/5 already
observed, thirty runs still leave a 4% rate looking clean about a third of the
time. Non-zero counts are solid; zeros are "not seen", never "does not happen".

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_final_rates.py
"""

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
RUNS = 30
SHAPES = ((17, 64), (19, 64), (10, 128), (12, 128), (20, 64), (64, 64), (1024, 64))


@triton.jit
def q_arm(q, src, table, tiles, V: tl.constexpr, HH: tl.constexpr, W: tl.constexpr):
    """Twenty lines, no fp8, no cache, no KV, no offsets. Just the Q shape."""
    pid = tl.program_id(0).to(tl.int64)
    tok = pid // tiles
    rows = tok * (tiles * HH) + (pid % tiles) * HH + tl.arange(0, HH)
    col = tl.arange(0, W)
    blk = tl.load(q + rows[:, None] * W + col[None, :]).to(tl.float32)
    rs = tl.rsqrt(tl.sum(blk * blk, axis=1) / W + 1e-6)
    blk = blk * rs[:, None]
    tl.store(q + rows[:, None] * W + col[None, :],
             blk.to(tl.bfloat16), mask=col[None, :] < W - V)
    p = tl.load(src + tok)
    half = tl.arange(0, V // 2)
    c = tl.load(table + p * V + half)
    s = tl.load(table + p * V + V // 2 + half)
    po = (rows[:, None, None] * W + (W - V)
          + half[None, :, None] * 2 + tl.arange(0, 2)[None, None, :])
    pair = tl.load(q + po).to(tl.float32)
    e, o = tl.split(pair)
    e, o = e * rs[:, None], o * rs[:, None]
    tl.store(q + po, tl.join(e * c[None, :] - o * s[None, :],
                             e * s[None, :] + o * c[None, :]).to(tl.bfloat16))


def main():
    import flaggems_vllm

    fn = flaggems_vllm.runtime.torch_device_fn
    dev = flaggems_vllm.device
    name = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    override = getattr(flaggems_vllm, name)

    def counted(make, run):
        """Runs are counted independently; a raise is reported, not fatal."""
        try:
            state = make()
        except Exception as e:
            return None, "setup: {}".format(str(e).strip().splitlines()[:1])
        try:
            ref = run(state)
            bad, worst, where = 0, 0, ""
            for _ in range(RUNS):
                g = run(state)
                m = g != ref
                d = int(m.sum())
                if d:
                    bad += 1
                    if d > worst:
                        worst = d
                        tk = m.any(-1).any(-1).nonzero().flatten() if g.dim() == 3 \
                            else m.any(-1).nonzero().flatten()
                        where = "{}..{}".format(int(tk.min()), int(tk.max()))
                del g, m
            del ref
            return (bad, worst, where), None
        except Exception as e:
            lines = [x for x in str(e).strip().splitlines() if x.strip()]
            return None, (lines[0][:44] if lines else type(e).__name__)
        finally:
            fn.empty_cache()

    print("  {:>7} {:>5} {:>7} {:>16} {:>16} {:>12}".format(
        "tokens", "heads", "grid", "override /30", "standalone /30", "worst rows"))
    print("  " + "-" * 72)

    ov_hits, st_hits = 0, 0
    for n, h in SHAPES:
        tiles, HH = h // 32, 32
        total = n * tiles + n

        def mk_real():
            torch.manual_seed(0)
            q = torch.randn(n, h, HEAD_DIM, dtype=torch.bfloat16, device=dev)
            kv = torch.randn(n, HEAD_DIM, dtype=torch.bfloat16, device=dev)
            pos = torch.arange(n, dtype=torch.int64, device=dev)
            inv = 1.0 / (10000.0 ** (torch.arange(0, ROPE_DIM, 2,
                                                  dtype=torch.float32,
                                                  device=dev) / ROPE_DIM))
            t = torch.arange(max(4096, n), dtype=torch.float32, device=dev)
            f = torch.einsum("i,j->ij", t, inv)
            cs = torch.cat((f.cos(), f.sin()), dim=-1)
            nb = (n + 63) // 64 + 1
            slot = torch.arange(n, dtype=torch.int64, device=dev)
            kc = torch.zeros(nb, 64 * HEAD_BYTES, dtype=torch.uint8, device=dev)
            return q, kv, kc, slot, pos, cs

        def run_real(st):
            q, kv, kc, slot, pos, cs = st
            qq, kk = q.clone(), kc.clone()
            override(qq, kv, kk, slot, pos, cs, EPS, 64)
            fn.synchronize()
            return qq

        def mk_std():
            torch.manual_seed(0)
            q = torch.randn(n * tiles * HH, HEAD_DIM, dtype=torch.bfloat16,
                            device=dev)
            src = torch.arange(n, dtype=torch.int64, device=dev)
            table = torch.randn(max(n, 8), ROPE_DIM, dtype=torch.float32, device=dev)
            return q, src, table

        def run_std(st):
            q0, src, table = st
            q = q0.clone()
            q_arm[(n * tiles,)](q, src, table, tiles, ROPE_DIM, HH, HEAD_DIM,
                                num_warps=1, num_stages=1)
            fn.synchronize()
            return q

        ov, ov_err = counted(mk_real, run_real)
        st, st_err = counted(mk_std, run_std)
        if ov and ov[0]:
            ov_hits += 1
        if st and st[0]:
            st_hits += 1
        print("  {:>7} {:>5} {:>7} {:>16} {:>16} {:>12}".format(
            n, h, total,
            "{}/{}".format(ov[0], RUNS) if ov else (ov_err or "err"),
            "{}/{}".format(st[0], RUNS) if st else (st_err or "err"),
            ov[2] if ov and ov[0] else ""))

    print()
    print("  shapes where the override was seen non-deterministic : {}".format(ov_hits))
    print("  shapes where the standalone kernel was               : {}".format(st_hits))
    print()
    if ov_hits and st_hits:
        print("BOTH. The 20-line kernel above reproduces it with no fp8, no paged")
        print("cache, no KV arm and nothing from this repository, so it is what")
        print("goes to the vendor. Editing the override cannot be the answer to a")
        print("defect that a kernel this small also shows.")
        print("\n[RESULT] STANDALONE_REPRODUCES")
    elif ov_hits:
        print("Only the shipped override showed it in this run. The standalone")
        print("kernel is not yet a reproducer -- grow it toward the real one, one")
        print("difference at a time, rather than sending it as is.")
        print("\n[RESULT] OVERRIDE_ONLY")
    else:
        print("Neither showed it in {} runs, which contradicts 19-of-19 at 17x64")
        print("earlier in this session. Something differs between then and now --")
        print("find it before writing any of this up.".format(RUNS))
        print("\n[RESULT] NOTHING_SEEN")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
