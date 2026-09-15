"""A/B latency of the launch-padding fix against the PR head, interleaved in one process.

The fix adds at most vector_core_num - 1 no-op programs to the last launch and
shortens the launch step from 65535 to 65520. Neither should cost anything that
matters, but decode shapes on this backend are launch-bound, so it is measured
rather than assumed.

OLD is the override file as of the PR head before the fix (git ref below), loaded
from `git show` into its own module; NEW is the installed file. The two kernels
have different source, so they compile and cache separately. For each shape the
two alternate for ROUNDS rounds and the median of each side is reported, so both
see the same device state and thermal drift.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_bench_pad.py
"""

import importlib
import importlib.util
import os
import statistics
import subprocess
import sys
import tempfile
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
import triton  # noqa: E402

OLD_REF = os.environ.get("OLD_REF", "c009054")
REL = "src/flaggems_vllm/runtime/backend/_ascend/fused/fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py"
W, V, HEAD_BYTES, EPS = 512, 64, 584, 1e-6
ROUNDS = 3
TOKENS = (1, 4, 17, 64, 1024, 2048, 8192, 32768, 65536, 98304, 131072)


def load_old():
    src = subprocess.run(["git", "-C", REPO, "show", "{}:{}".format(OLD_REF, REL)],
                         capture_output=True, text=True, check=True).stdout
    assert "launch_group_size" not in src, "OLD_REF already contains the fix"
    path = os.path.join(tempfile.mkdtemp(), "fv4_old_override.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("fv4_old_override", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import flaggems_vllm

    fn = flaggems_vllm.runtime.torch_device_fn
    dev = flaggems_vllm.device
    new = importlib.import_module("flaggems_vllm.runtime.backend._ascend.fused"
                                  ".fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")
    assert hasattr(new, "launch_group_size"), "installed override does not have the fix"
    old = load_old()
    impls = {"old": old.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
             "new": new.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert}

    print("OLD = {}:{}".format(OLD_REF, REL.split("/")[-1]))
    print("\n  {:>7} {:>5} {:>7} {:>12} {:>12} {:>9}".format(
        "tokens", "heads", "total", "old ms", "new ms", "new/old"))
    print("  " + "-" * 58)
    worst = []
    for h in (64, 128):
        for n in TOKENS:
            try:
                torch.manual_seed(0)
                q = torch.randn(n, h, W, dtype=torch.bfloat16, device=dev)
                kv = torch.randn(n, W, dtype=torch.bfloat16, device=dev)
                pos = torch.arange(n, dtype=torch.int64, device=dev)
                inv = 1.0 / (10000.0 ** (torch.arange(0, V, 2, dtype=torch.float32,
                                                      device=dev) / V))
                t = torch.arange(max(4096, n), dtype=torch.float32, device=dev)
                f = torch.einsum("i,j->ij", t, inv)
                cs = torch.cat((f.cos(), f.sin()), dim=-1)
                slot = torch.arange(n, dtype=torch.int64, device=dev)
                kc = torch.zeros((n + 63) // 64 + 1, 64 * HEAD_BYTES, dtype=torch.uint8,
                                 device=dev)
                for name in impls:                         # compile both first
                    impls[name](q, kv, kc, slot, pos, cs, EPS, 64)
                fn.synchronize()
                times = {"old": [], "new": []}
                for _ in range(ROUNDS):
                    for name in ("old", "new"):
                        ms = triton.testing.do_bench(
                            lambda: impls[name](q, kv, kc, slot, pos, cs, EPS, 64),
                            warmup=25, rep=200, return_mode="median")
                        times[name].append(ms)
                o, nw = statistics.median(times["old"]), statistics.median(times["new"])
                tiles = h // new.q_heads_per_program(h)
                ratio = nw / o
                worst.append((ratio, n, h))
                print("  {:>7} {:>5} {:>7} {:>12.4f} {:>12.4f} {:>9.3f}".format(
                    n, h, n * tiles + n, o, nw, ratio), flush=True)
                del q, kv, kc, cs
            except Exception as e:
                lines = [x for x in str(e).splitlines() if x.strip()]
                print("  {:>7} {:>5}   ERROR {}".format(
                    n, h, (lines[0] if lines else type(e).__name__)[:50]))
            fn.empty_cache()

    if worst:
        worst.sort(reverse=True)
        print("\n  slowest new/old: " + ", ".join(
            "{:.3f} at {}x{}".format(r, n, h) for r, n, h in worst[:3]))
    print("\n[RESULT] BENCH_DONE")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
