"""Attribute the >40-program slowdown: kernel change, padding, or host wrapper.

The noise-floor A/B established a real cost only above 40 programs: new/old
1.118-1.180 at 17x64, 64x64, 17x128 and 64x128 against an A/A floor of
0.995-1.043, while <=40-program shapes read 1.014-1.036. 64x128 is 320 programs,
pads nothing, and is still 1.144, so padding programs alone cannot be it.

That probe's decomposition was wrong: its `newk` launched the new kernel
directly and came out ~40% FASTER than old, because it also skipped the whole
Python wrapper (~27 us). do_bench on this part DOES include host time. So each
comparison here is like-for-like:

  oldk  : old kernel launched directly, old grid
  newk  : new kernel launched directly, same grid        newk - oldk  = kernel change
  newkp : new kernel launched directly, padded grid      newkp - newk = padding
  old   : old public function                            (new - newkp) - (old - oldk)
  new   : new public function                                         = host wrapper change
  old2  : old again                                       old2 - old   = floor

Six slots, order rotated cyclically across 12 rounds, every round printed.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_bench_decompose.py
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
ROUNDS = 12
SHAPES = ((4, 64), (17, 64), (64, 128), (256, 64), (1024, 64))
SLOTS = ("old", "old2", "oldk", "newk", "newkp", "new")


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
    new = importlib.import_module("flaggems_vllm.runtime.backend._ascend.fused"
                                  ".fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")
    assert hasattr(new, "launch_group_size")
    old = load_old()
    group = new.launch_group_size(torch.npu.current_device())

    table = []
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
            padded = total if total <= group else -(-total // group) * group
            st = kc.stride(0)

            def oldk():
                old.fused_qnorm_rope_kv_insert_kernel[(total,)](
                    q, kv, kc, kcb, slot, pos, cs, EPS, 64, h, st, 0, qp, tiles, Hp,
                    num_warps=1, num_stages=1)

            def newk():
                new.fused_qnorm_rope_kv_insert_kernel[(total,)](
                    q, kv, kc, kcb, slot, pos, cs, EPS, 64, h, st, 0, qp, total, tiles, Hp,
                    num_warps=1, num_stages=1)

            def newkp():
                new.fused_qnorm_rope_kv_insert_kernel[(padded,)](
                    q, kv, kc, kcb, slot, pos, cs, EPS, 64, h, st, 0, qp, total, tiles, Hp,
                    num_warps=1, num_stages=1)

            call = {
                "old": lambda: old.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                    q, kv, kc, slot, pos, cs, EPS, 64),
                "new": lambda: new.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                    q, kv, kc, slot, pos, cs, EPS, 64),
                "oldk": oldk, "newk": newk, "newkp": newkp,
            }
            call["old2"] = call["old"]
            for s in ("old", "new", "oldk", "newk", "newkp"):
                call[s]()
            fn.synchronize()

            per = {s: [] for s in SLOTS}
            for r in range(ROUNDS):
                order = SLOTS[r % len(SLOTS):] + SLOTS[:r % len(SLOTS)]
                for s in order:
                    per[s].append(triton.testing.do_bench(call[s], warmup=25, rep=300,
                                                          return_mode="median") * 1000)
            m = {s: statistics.median(per[s]) for s in SLOTS}
            print("\n{}x{}  total {} programs, padded {}".format(n, h, total, padded))
            for s in SLOTS:
                print("  {:<5} {:>8.2f} us   {}".format(
                    s, m[s], " ".join("{:.1f}".format(x) for x in per[s])))
            row = dict(shape="{}x{}".format(n, h), total=total, padded=padded,
                       floor=m["old2"] - m["old"], total_d=m["new"] - m["old"],
                       kernel=m["newk"] - m["oldk"], pad=m["newkp"] - m["newk"],
                       host=(m["new"] - m["newkp"]) - (m["old"] - m["oldk"]),
                       old=m["old"])
            table.append(row)
            del q, kv, kc, cs
        except Exception as e:
            lines = [x for x in str(e).splitlines() if x.strip()]
            print("\n{}x{}  ERROR {}".format(n, h, (lines[0] if lines else type(e).__name__)[:60]))
        fn.empty_cache()

    print("\nDeltas in us (medians of {} rounds):".format(ROUNDS))
    print("{:>9} {:>6} {:>7} {:>8} {:>9} {:>9} {:>8} {:>8}".format(
        "shape", "total", "padded", "old us", "new-old", "kernel", "padding", "host"))
    for r in table:
        print("{:>9} {:>6} {:>7} {:>8.1f} {:>9.1f} {:>9.1f} {:>8.1f} {:>8.1f}   (floor {:+.1f})".format(
            r["shape"], r["total"], r["padded"], r["old"], r["total_d"], r["kernel"],
            r["pad"], r["host"], r["floor"]))
    print("\n[RESULT] DECOMPOSE_DONE")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
