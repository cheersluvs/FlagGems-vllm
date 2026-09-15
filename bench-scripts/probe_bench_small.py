"""Is the small-shape slowdown real? Noise floor first, then decompose.

The first A/B showed new/old 1.03-1.16 at 1-64 tokens and 0.994-1.001 from 1024
up. Those calls are 65-78 us, below the ~100 us floor under which ratios on
these parts stop being quotable; the run had 3 rounds, a FIXED old-then-new
order, and no A/A control. It also does not fit padding: 64x128 is 320
programs, a multiple of 40, pads nothing, and read 1.147.

This follows the noise-floor protocol:
  * A/A control: the old callable timed in two slots, so their ratio IS the floor
  * order rotated across rounds (4x4 Latin square), so no slot is always first
  * 8 rounds, every round printed
and decomposes the candidate into what changed:
  old   : PR head override
  old2  : the same callable again (floor)
  new   : fixed override through its public function (padding + new kernel)
  newk  : the NEW kernel launched with the OLD grid (no padding), isolating the
          kernel change itself -- one more runtime argument and the `elif`
do_bench excludes most host launch overhead on Ascend, so host-side lines in the
fix should be invisible here; if new is slower, newk says whether the kernel is why.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_bench_small.py
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
ROUNDS = 8
SHAPES = ((1, 64), (4, 64), (17, 64), (64, 64), (1, 128), (4, 128), (17, 128), (64, 128))
SLOTS = ("old", "old2", "new", "newk")
ORDERS = (SLOTS, ("old2", "newk", "old", "new"), ("new", "old", "newk", "old2"),
          ("newk", "new", "old2", "old"))


def load_old():
    src = subprocess.run(["git", "-C", REPO, "show", "{}:{}".format(OLD_REF, REL)],
                         capture_output=True, text=True, check=True).stdout
    assert "launch_group_size" not in src
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
    try:
        out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True,
                             timeout=20).stdout
        busy = [x for x in out.splitlines() if "python" in x.lower() or "No running" in x]
        print("npu-smi process lines: " + (" | ".join(x.strip() for x in busy[:4]) or "(none matched)"))
    except Exception as e:
        print("npu-smi unavailable: {}".format(e))

    new = importlib.import_module("flaggems_vllm.runtime.backend._ascend.fused"
                                  ".fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")
    assert hasattr(new, "launch_group_size")
    old = load_old()

    summary = []
    for n, h in SHAPES:
        try:
            torch.manual_seed(0)
            q = torch.randn(n, h, W, dtype=torch.bfloat16, device=dev)
            kv = torch.randn(n, W, dtype=torch.bfloat16, device=dev)
            pos = torch.arange(n, dtype=torch.int64, device=dev)
            inv = 1.0 / (10000.0 ** (torch.arange(0, V, 2, dtype=torch.float32, device=dev) / V))
            t = torch.arange(4096, dtype=torch.float32, device=dev)
            f = torch.einsum("i,j->ij", t, inv)
            cs = torch.cat((f.cos(), f.sin()), dim=-1)
            slot = torch.arange(n, dtype=torch.int64, device=dev)
            kc = torch.zeros((n + 63) // 64 + 1, 64 * HEAD_BYTES, dtype=torch.uint8, device=dev)
            kcb = kc.view(torch.bfloat16)
            Hp = new.q_heads_per_program(h)
            tiles = h // Hp
            qp, total = n * tiles, n * tiles + n

            def newk():
                new.fused_qnorm_rope_kv_insert_kernel[(total,)](
                    q, kv, kc, kcb, slot, pos, cs, EPS, 64, h, kc.stride(0), 0, qp,
                    total, tiles, Hp, num_warps=1, num_stages=1)

            call = {
                "old": lambda: old.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                    q, kv, kc, slot, pos, cs, EPS, 64),
                "new": lambda: new.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                    q, kv, kc, slot, pos, cs, EPS, 64),
                "newk": newk,
            }
            call["old2"] = call["old"]
            for s in ("old", "new", "newk"):
                call[s]()
            fn.synchronize()

            per = {s: [] for s in SLOTS}
            for r in range(ROUNDS):
                for s in ORDERS[r % len(ORDERS)]:
                    per[s].append(triton.testing.do_bench(call[s], warmup=25, rep=300,
                                                          return_mode="median") * 1000)
            med = {s: statistics.median(per[s]) for s in SLOTS}
            print("\n{}x{}  (total {} programs, new pads to {})".format(
                n, h, total, total if total <= 40 else -(-total // 40) * 40))
            for s in SLOTS:
                print("  {:<5} median {:>7.2f} us  rounds {}".format(
                    s, med[s], " ".join("{:.1f}".format(x) for x in per[s])))
            spread = (max(per["old"]) - min(per["old"])) / med["old"]
            row = (n, h, med["old2"] / med["old"], med["new"] / med["old"],
                   med["newk"] / med["old"], spread)
            summary.append(row)
            print("  old2/old {:.3f}   new/old {:.3f}   newk/old {:.3f}   old spread {:.1%}".format(*row[2:]))
            del q, kv, kc, cs
        except Exception as e:
            lines = [x for x in str(e).splitlines() if x.strip()]
            print("\n{}x{}  ERROR {}".format(n, h, (lines[0] if lines else type(e).__name__)[:60]))
        fn.empty_cache()

    print("\n{:>7} {:>5} {:>9} {:>8} {:>9} {:>11}".format(
        "tokens", "heads", "old2/old", "new/old", "newk/old", "old spread"))
    for n, h, aa, nw, nk, sp in summary:
        print("{:>7} {:>5} {:>9.3f} {:>8.3f} {:>9.3f} {:>10.1%}".format(n, h, aa, nw, nk, sp))
    print("""
Reading it:
  new/old inside the old2/old deviation and the old spread -> no measurable cost
  new/old beyond both, newk/old about the same             -> the kernel change
                                                              (extra arg / elif) costs it
  new/old beyond both, newk/old near 1                     -> padding or host path costs it
""")
    print("[RESULT] BENCH_SMALL_DONE")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
