"""Verify the launch-padding fix through the public API, against a host reference.

The fix (branch ascend-pad-launch) keeps every launch at most one core group or a
whole number of groups, because the runtime otherwise re-executes program 0 2-7
times and this operator writes q in place. This checks the whole dispatch --
flaggems_vllm.<op>, not the kernel -- at the decode shapes that failed, around
the group boundary, and at sizes that need padding in a multi-launch sequence.

Q is scored against a float32 host reference of the Q path (RMSNorm without
weight, GPT-J RoPE with cos/sin from the cache), counting gross errors
(rel > 1e-2). k_cache is checked for run-to-run identity. 0 gross errors on every
run is the pass condition; before the fix, 17x64 failed 9 runs in 10.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_verify_pad.py
"""

import importlib
import os
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402

W, V, HEAD_BYTES, EPS, GROSS = 512, 64, 584, 1e-6, 1e-2
SHAPES = [(1, 64), (4, 64), (13, 64), (14, 64), (17, 64), (19, 64), (20, 64),
          (21, 64), (27, 64), (12, 128), (15, 128), (64, 64), (100, 128),
          (1024, 64), (1024, 128), (8192, 64), (21846, 64)]


def host_ref(q0, cs, n, h):
    q = q0.reshape(-1, W).cpu().clone()
    blk = q.float()
    rs = torch.rsqrt((blk * blk).sum(1) / W + EPS)
    tok = torch.arange(n * h) // h
    c, s = cs.cpu()[tok, :V // 2], cs.cpu()[tok, V // 2:]
    pair = blk[:, W - V:].reshape(-1, V // 2, 2)
    e, o = pair[..., 0] * rs[:, None], pair[..., 1] * rs[:, None]
    q[:, :W - V] = (blk[:, :W - V] * rs[:, None]).to(torch.bfloat16)
    q[:, W - V:] = torch.stack((e * c - o * s, e * s + o * c), -1) \
        .reshape(-1, V).to(torch.bfloat16)
    return q


def main():
    import flaggems_vllm

    mod = importlib.import_module("flaggems_vllm.runtime.backend._ascend.fused"
                                  ".fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")
    if not hasattr(mod, "launch_group_size"):
        print("the installed override does not have the fix (no launch_group_size)")
        print("\n[RESULT] FIX_NOT_INSTALLED")
        return
    op = flaggems_vllm.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    fn = flaggems_vllm.runtime.torch_device_fn
    dev = flaggems_vllm.device
    group = mod.launch_group_size(torch.npu.current_device())
    print("launch group size = {}   launch step = {}".format(
        group, mod.MAX_PROGRAMS_PER_LAUNCH // group * group))

    print("\n  {:>11} {:>7} {:>9} {:<44} {}".format(
        "shape", "total", "launches", "gross errors in Q per run", "k_cache stable"))
    failed = []
    for n, h in SHAPES:
        runs = 10 if n <= 1024 else 3
        try:
            tiles = h // mod.q_heads_per_program(h)
            total = n * tiles + n
            step = mod.MAX_PROGRAMS_PER_LAUNCH // group * group
            torch.manual_seed(0)
            q0 = torch.randn(n, h, W, dtype=torch.bfloat16, device=dev)
            kv = torch.randn(n, W, dtype=torch.bfloat16, device=dev)
            pos = torch.arange(n, dtype=torch.int64, device=dev)
            inv = 1.0 / (10000.0 ** (torch.arange(0, V, 2, dtype=torch.float32,
                                                  device=dev) / V))
            t = torch.arange(max(4096, n), dtype=torch.float32, device=dev)
            f = torch.einsum("i,j->ij", t, inv)
            cs = torch.cat((f.cos(), f.sin()), dim=-1)
            slot = torch.arange(n, dtype=torch.int64, device=dev)
            kc0 = torch.zeros((n + 63) // 64 + 1, 64 * HEAD_BYTES, dtype=torch.uint8,
                              device=dev)
            ref = host_ref(q0, cs, n, h).float()
            grosses, kc_first, kc_ok = [], None, True
            for _ in range(runs):
                q, kc = q0.clone(), kc0.clone()
                op(q, kv, kc, slot, pos, cs, EPS, 64)
                fn.synchronize()
                a = q.reshape(-1, W).cpu().float()
                grosses.append(int(((a - ref).abs() / ref.abs().clamp(min=1e-6) > GROSS).sum()))
                kcc = kc.cpu()
                if kc_first is None:
                    kc_first = kcc
                else:
                    kc_ok = kc_ok and torch.equal(kc_first, kcc)
                del q, kc, a, kcc
            if any(grosses) or not kc_ok:
                failed.append((n, h))
            print("  {:>11} {:>7} {:>9} {:<44} {}".format(
                "{}x{}".format(n, h), total, -(-total // step), str(grosses),
                "yes" if kc_ok else "NO"), flush=True)
            del q0, kv, kc0, ref, kc_first
        except Exception as e:
            lines = [x for x in str(e).splitlines() if x.strip()]
            print("  {:>11}  ERROR {}".format("{}x{}".format(n, h),
                                              (lines[0] if lines else type(e).__name__)[:60]))
            failed.append((n, h))
        fn.empty_cache()

    print()
    if failed:
        print("NOT FIXED at: {}".format(failed))
        print("\n[RESULT] VERIFY_FAILED")
    else:
        print("Q matches the host reference with zero gross errors on every run at")
        print("every shape, and k_cache is identical across runs. Next: the full test")
        print("suite, twice.")
        print("\n[RESULT] VERIFY_PASSED")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
