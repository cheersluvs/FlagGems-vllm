"""The small, constant residual after the fix: rounding in the reference, or a defect?

After padding, decode shapes are clean, but larger shapes keep a few gross
errors that are IDENTICAL every run and scale with element count: 1 at
100x128, 1 at 1024x64, 2 at 1024x128, 5 at 8192x64, 17 at 21846x64. That does
not look like program-0 re-execution (varying, token 0, hundreds). A plausible
benign cause: bfloat16 has 7 fraction bits, so neighbouring values are 0.78% to
1.56% apart -- above the 1e-2 gross threshold -- and device fp32 arithmetic
(fused multiplies, a different tl.sum order) can land one representable value
away from host torch.

This checks instead of assuming. For each gross element: where it is (token 0
or not), how many representable bf16 steps separate device and reference, and
which side a float64 reference rounds to. One step apart and the float64
reference siding with either is rounding. Anything else is not.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_residual.py
"""

import os
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

W, V, HEAD_BYTES, EPS, GROSS = 512, 64, 584, 1e-6, 1e-2


def ref_rows(qrows, cs_rows, dt):
    blk = qrows.to(dt)
    rs = torch.rsqrt((blk * blk).sum(1) / W + EPS)
    c, s = cs_rows[:, :V // 2].to(dt), cs_rows[:, V // 2:].to(dt)
    pair = blk[:, W - V:].reshape(-1, V // 2, 2)
    e, o = pair[..., 0] * rs[:, None], pair[..., 1] * rs[:, None]
    out = torch.empty_like(qrows)
    out[:, :W - V] = (blk[:, :W - V] * rs[:, None]).to(torch.bfloat16)
    out[:, W - V:] = torch.stack((e * c - o * s, e * s + o * c), -1) \
        .reshape(-1, V).to(torch.bfloat16)
    return out


def steps(a, b):
    """Representable bf16 steps between two bf16 tensors (same sign assumed)."""
    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()


def main():
    import flaggems_vllm

    op = flaggems_vllm.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    fn = flaggems_vllm.runtime.torch_device_fn
    dev = flaggems_vllm.device
    verdict_ok = True
    for n, h in ((100, 128), (1024, 128), (8192, 64)):
        torch.manual_seed(0)
        q0 = torch.randn(n, h, W, dtype=torch.bfloat16, device=dev)
        kv = torch.randn(n, W, dtype=torch.bfloat16, device=dev)
        pos = torch.arange(n, dtype=torch.int64, device=dev)
        inv = 1.0 / (10000.0 ** (torch.arange(0, V, 2, dtype=torch.float32, device=dev) / V))
        t = torch.arange(max(4096, n), dtype=torch.float32, device=dev)
        f = torch.einsum("i,j->ij", t, inv)
        cs = torch.cat((f.cos(), f.sin()), dim=-1)
        slot = torch.arange(n, dtype=torch.int64, device=dev)
        kc = torch.zeros((n + 63) // 64 + 1, 64 * HEAD_BYTES, dtype=torch.uint8, device=dev)
        q = q0.clone()
        op(q, kv, kc, slot, pos, cs, EPS, 64)
        fn.synchronize()

        q0c = q0.reshape(-1, W).cpu()
        got = q.reshape(-1, W).cpu()
        tok = torch.arange(n * h) // h
        cs_c = cs.cpu()
        bad_rows, bad_cols = [], []
        chunk = 65536
        for r0 in range(0, n * h, chunk):
            r1 = min(r0 + chunk, n * h)
            ref = ref_rows(q0c[r0:r1], cs_c[tok[r0:r1]], torch.float32).float()
            a = got[r0:r1].float()
            g = (a - ref).abs() / ref.abs().clamp(min=1e-6) > GROSS
            idx = g.nonzero()
            bad_rows += (idx[:, 0] + r0).tolist()
            bad_cols += idx[:, 1].tolist()
        print("\n{}x{}: {} gross elements".format(n, h, len(bad_rows)))
        print("  {:>8} {:>5} {:>4} {:>13} {:>13} {:>13} {:>7} {:>7}".format(
            "token", "head", "dim", "device", "ref f32", "ref f64", "steps32", "steps64"))
        for r, cidx in zip(bad_rows[:20], bad_cols[:20]):
            row = q0c[r:r + 1]
            csr = cs_c[tok[r:r + 1]]
            r32 = ref_rows(row, csr, torch.float32)[0, cidx]
            r64 = ref_rows(row, csr, torch.float64)[0, cidx]
            d = got[r, cidx]
            s32 = int(steps(d.reshape(1), r32.reshape(1)))
            s64 = int(steps(d.reshape(1), r64.reshape(1)))
            if min(s32, s64) > 1 or int(tok[r]) == 0:
                verdict_ok = False
            print("  {:>8} {:>5} {:>4} {:>13.6g} {:>13.6g} {:>13.6g} {:>7} {:>7}".format(
                int(tok[r]), r % h, cidx, float(d), float(r32), float(r64), s32, s64))
        del q0, kv, kc, q, cs
        fn.empty_cache()

    print()
    if verdict_ok:
        print("Every residual element is at most one representable bf16 step from a")
        print("reference, and none is at token 0. That is rounding, not the defect;")
        print("the gross threshold is simply below bf16's own resolution.")
        print("\n[RESULT] RESIDUAL_IS_ROUNDING")
    else:
        print("Some residual element is more than one step from both references, or")
        print("sits at token 0. That is not rounding and needs looking at.")
        print("\n[RESULT] RESIDUAL_NOT_ROUNDING")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
