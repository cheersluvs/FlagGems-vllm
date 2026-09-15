"""Which host-side line costs the ~6-10 us? Timed without the device.

The like-for-like decomposition put the whole small-shape slowdown in the host
wrapper: at 64x128 the kernel change was -0.1 us, padding 0.0 us, host +10.2 us,
with the three direct-launch slots steady at 41.2 us. At 1024x64 the host delta
vanishes because the wrapper runs while the device is still busy.

The added host lines look like ~2-3 us in total, which does not match, so they
are measured rather than guessed. Both wrappers are timed with their kernel
replaced by a no-op stub (a module global, looked up at call time), so this is
pure Python/torch host cost with sub-microsecond resolution and no device noise.
Then each added line is timed alone.

    cd <repo> && PYTHONPATH=src:$PYTHONPATH python3 bench-scripts/probe_host_cost.py
"""

import importlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import timeit
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
import triton  # noqa: E402

OLD_REF = os.environ.get("OLD_REF", "c009054")
REL = "src/flaggems_vllm/runtime/backend/_ascend/fused/fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py"
W, V, HEAD_BYTES, EPS = 512, 64, 584, 1e-6
N = 20000


class Stub:
    def __getitem__(self, grid):
        return lambda *a, **k: None


def load_old():
    src = subprocess.run(["git", "-C", REPO, "show", "{}:{}".format(OLD_REF, REL)],
                         capture_output=True, text=True, check=True).stdout
    path = os.path.join(tempfile.mkdtemp(), "fv4_old_override.py")
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("fv4_old_override", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def us(stmt, g, n=N, repeat=5):
    return min(timeit.repeat(stmt, globals=g, number=n, repeat=repeat)) / n * 1e6


def main():
    import flaggems_vllm

    dev = flaggems_vllm.device
    new = importlib.import_module("flaggems_vllm.runtime.backend._ascend.fused"
                                  ".fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")
    old = load_old()
    real_new_k = new.fused_qnorm_rope_kv_insert_kernel
    new.fused_qnorm_rope_kv_insert_kernel = Stub()
    old.fused_qnorm_rope_kv_insert_kernel = Stub()
    try:
        print("  {:>9}  {:>10} {:>10} {:>8}".format("shape", "old us", "new us", "delta"))
        for n, h in ((17, 64), (64, 128), (1024, 64)):
            torch.manual_seed(0)
            q = torch.randn(n, h, W, dtype=torch.bfloat16, device=dev)
            kv = torch.randn(n, W, dtype=torch.bfloat16, device=dev)
            pos = torch.arange(n, dtype=torch.int64, device=dev)
            cs = torch.randn(4096, V, dtype=torch.float32, device=dev)
            slot = torch.arange(n, dtype=torch.int64, device=dev)
            kc = torch.zeros((n + 63) // 64 + 1, 64 * HEAD_BYTES, dtype=torch.uint8, device=dev)
            g = dict(old=old, new=new, q=q, kv=kv, kc=kc, slot=slot, pos=pos, cs=cs, EPS=EPS)
            o = us("old.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(q, kv, kc, slot, pos, cs, EPS, 64)", g)
            nw = us("new.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(q, kv, kc, slot, pos, cs, EPS, 64)", g)
            print("  {:>9}  {:>10.2f} {:>10.2f} {:>+8.2f}".format("{}x{}".format(n, h), o, nw, nw - o))

        print("\n  each added host line, alone:")
        g = dict(q=q, new=new, triton=triton, torch=torch)
        for label, stmt in (
            ("q.device", "q.device"),
            ("q.device.index", "q.device.index"),
            ("torch.npu.current_device()", "torch.npu.current_device()"),
            ("launch_group_size(0) (cached)", "new.launch_group_size(0)"),
            ("MAX // 40 * 40", "new.MAX_PROGRAMS_PER_LAUNCH // 40 * 40"),
            ("triton.cdiv(51, 40) * 40", "triton.cdiv(51, 40) * 40"),
            ("-(-51 // 40) * 40", "-(-51 // 40) * 40"),
        ):
            print("    {:<32} {:>8.3f} us".format(label, us(stmt, g, n=200000)))
        print("\n  q.device.index value: {!r}".format(q.device.index))
        print("  launch_group_size cache: {}".format(new.launch_group_size.cache_info()))
    finally:
        new.fused_qnorm_rope_kv_insert_kernel = real_new_k
    print("\n[RESULT] HOST_COST_DONE")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
