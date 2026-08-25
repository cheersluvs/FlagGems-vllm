"""The 22-shape C550 table against the rebuilt mcoplib 0.4.6 baseline.

The baseline is MetaX's 0.4.6 source with MetaX's own 0.4.9 launch fix applied,
rebuilt locally and bound under `mcoplib_rebuilt`. Run
`rebuild_mcoplib_046.py` first; this reuses the extension it built (ninja is a
no-op once it exists).

**Say what this baseline is.** Vendor source plus the vendor's own later fix,
rebuilt here. NOT a vendor-published wheel -- none exists to get, and the wheel
that is installed cannot launch this kernel at all. That is a weaker provenance
than T-Head, where the vendor's own published package runs with no patch, and a
table putting the two side by side has to say so.

TWO THINGS THIS MEASURES BEYOND THE RATIO.

1. **The magnitude of the q difference, not just its count.** The launch test
   reported 1 / 119 / 0 differing bfloat16 elements with k_cache bit-identical.
   A count cannot support "within tolerance"; the relative difference can. The
   repo's own test compares q at rtol=atol=1e-2, so that is the bar used here.

2. **Two runs, with the spread.** C550 has a clock ramp (unlike T-Head), so a
   cold first sweep reads low. do_bench warms up per shape, but a global warm-up
   runs first anyway and both sweeps are reported so a drifting cell is visible
   rather than averaged away.

    REPO=/path/to/FlagGems-vllm python3 run_c550_table.py
"""

import importlib
import os
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
WORK = "/tmp/mcoplib_rebuild"
SRC = os.path.join(WORK, "mcoplib")
CU = os.path.join(SRC, "op/vllm/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu")
BINDING = os.path.join(WORK, "binding.cpp")

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
from torch.utils.cpp_extension import load  # noqa: E402


def main():
    if not os.path.isfile(CU):
        print("找不到重建的源码 —— 先跑 rebuild_mcoplib_046.py")
        print("\n[RESULT] NOT_BUILT")
        return
    # is_python_module=False:绑定是 TORCH_LIBRARY,注册进调度器,没有 PyInit_
    load(name="mcoplib_rebuilt", sources=[CU, BINDING],
         extra_include_paths=[os.path.join(SRC, "op", "vllm")],
         is_python_module=False, verbose=False)
    ref = torch.ops.mcoplib_rebuilt.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert

    from benchmark import conftest as cf
    from benchmark import consts

    assert cf.Config is None
    cf.Config = cf.BenchConfig()
    cf.Config.mode = consts.BenchMode.KERNEL
    cf.Config.bench_level = consts.BenchLevel.CORE
    cf.Config.query = False

    import flaggems_vllm

    mod = importlib.import_module(
        "benchmark.test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    )
    cls = mod.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark

    # ---- 1. 差异幅度,不只是个数 -----------------------------------------
    print("=" * 78)
    print("  C550 — FlagGems vs 重建的 mcoplib 0.4.6(厂商源码 + 厂商自己的修复)")
    print("=" * 78)
    print("\n### 差异幅度(个数不足以支撑“在容差内”这句话)\n")
    print("  {:<12} {:>10} {:>14} {:>14} {:>16}".format(
        "shape", "q 差异数", "最大相对差", "k_cache 差异", "rtol=1e-2 内"))
    ok = True
    for n, h in ((64, 64), (1024, 64), (17, 128), (2048, 128)):
        p = mod.TestParam(n, h, num_tokens_insert=n, block_size=64,
                          max_pos=4096, eps=1e-6)
        for inp in cls.make_input(p):
            q, kv, kc, slot, pos, cs, eps, bs = inp
            q2, kc2 = q.clone(), kc.clone()
            ref(q, kv, kc, slot, pos, cs, eps, bs)
            flaggems_vllm.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
                q2, kv, kc2, slot, pos, cs, eps, bs)
            torch.cuda.synchronize()
            a, b = q.cpu().float(), q2.cpu().float()
            d = a != b
            dq = int(d.sum())
            # 位距在零附近是错的度量:1e-40 和 2e-40 相隔很多个 ULP 却毫无差别。
            # 用相对差,并按仓库自己的容差判定。
            rel = 0.0
            if dq:
                rel = float(((a - b).abs() / b.abs().clamp(min=1e-30))[d].max())
            close = torch.allclose(a, b, rtol=1e-2, atol=1e-2)
            dc = int((kc.cpu() != kc2.cpu()).sum())
            ok = ok and dc == 0 and close
            print("  {:<12} {:>10} {:>14.3e} {:>14} {:>16}".format(
                "{}x{}".format(n, h), dq, rel, dc, str(close)))
            break
        torch.cuda.empty_cache()
    if not ok:
        print("\n  k_cache 不一致或 q 超出仓库容差 —— 比值会是在比不同的计算。")
        print("\n[RESULT] MISMATCH")
        return
    print("\n  k_cache 逐字节一致,q 在仓库自己的 rtol=atol=1e-2 内。\n")

    # ---- 2. 全局预热:这张卡有时钟爬坡 ------------------------------------
    p = mod.TestParam(8192, 64, num_tokens_insert=8192, block_size=64,
                      max_pos=4096, eps=1e-6)
    for inp in cls.make_input(p):
        for _ in range(30):
            flaggems_vllm.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(*inp)
        torch.cuda.synchronize()
        break
    torch.cuda.empty_cache()
    print("  已预热(C550 有时钟爬坡,冷跑第一轮会偏低)\n")

    # ---- 3. 两轮 harness --------------------------------------------------
    real_make = cls.make_input

    def make_input_freeing(param):
        torch.cuda.empty_cache()
        yield from real_make(param)

    cls.make_input = staticmethod(make_input_freeing)

    import benchmark.base as base
    base.pytest.fail = lambda *a, **k: None

    def one(label):
        bench = cls()
        bench.torch_op = ref            # 重建的基线绑进空槽
        real_latency = bench.get_latency
        rows, pending = [], {}

        def timed(op, *args, **kwargs):
            q = args[0]
            key = (int(q.shape[0]), int(q.shape[1]))
            ms = real_latency(op, *args, **kwargs)
            if key in pending:
                b = pending.pop(key)
                rows.append((key[0], key[1], b, ms))
                print("    {:>7} {:>5}  vendor {:>9.4f}  gems {:>9.4f}  {:>7.3f}x"
                      .format(key[0], key[1], b, ms, b / ms), flush=True)
            else:
                pending[key] = ms
            return ms

        bench.get_latency = timed
        print("=== 第 {} 轮 ===".format(label), flush=True)
        bench.run()
        return rows

    r1 = one(1)
    r2 = one(2)

    m2 = {(n, h): (b, g) for n, h, b, g in r2}
    print("\n\n{:>7} {:>5} | {:>10} {:>10} {:>8} | {:>8} | {:>7}".format(
        "tokens", "heads", "vendor①", "gems①", "①", "②", "偏差"))
    print("-" * 70)
    worst = 0.0
    for n, h, b1, g1 in r1:
        s1 = b1 / g1
        if (n, h) not in m2:
            print("{:>7} {:>5} | {:>10.4f} {:>10.4f} {:>7.3f}x | {:>8} |"
                  .format(n, h, b1, g1, s1, "缺"))
            continue
        b2, g2 = m2[(n, h)]
        s2 = b2 / g2
        dd = abs(s1 - s2) / max(s1, s2)
        worst = max(worst, dd)
        flag = "  <- 偏差 >5%" if dd > 0.05 else ""
        print("{:>7} {:>5} | {:>10.4f} {:>10.4f} {:>7.3f}x | {:>7.3f}x | {:>6.1%}{}"
              .format(n, h, b1, g1, s1, s2, dd, flag))
    print("-" * 70)
    print("两轮最大偏差 {:.1%}".format(worst))
    print("\n加速比 = 重建的 mcoplib 0.4.6 / FlagGems,--mode kernel --level core")
    print("基线出处:厂商源码 + 厂商自己在 0.4.9 的修复,本地重建 —— 不是厂商发布的 wheel")
    print("\n[RESULT] C550_TABLE_OK")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
