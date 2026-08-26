"""S5000 全 22 形状:基线用上游原样的启动配置(不传 num_warps)。

对照的是 PR 现在那张表 —— 基线跑在自己实测最优 `num_warps=1`,给出 1.20x。
本脚本给出「照上游原样用」会得到什么,已测得单 kernel 上是 2.23x
(465 vs 1036 GB/s)。两张表都要有,PR 才能说清为什么选了后者。

**「原样」的边界,说清楚免得误读。** 指的是**启动配置**:上游不传 num_warps
(v0.27.1 与 main 的第 135 行都没有,也没有 @triton.autotune),于是取 Triton
默认的 4。两个可移植性补丁必须留着,否则跑不完:

  * int32 溢出(vllm#52416,无人接):token_idx 是 int32,q 超过 2^31 元素就
    回绕。benchmark 的形状表到 131072x128,不修则大形状必死。
  * 存储竞争(vllm#52415,PR #52582 未合并):影响数值不影响计时,但既然
    已经在接线里,就不动它。

**不信任任何标志位。** 记录在案的一次事故:接线脚本的幂等检查只判断补丁在不在、
不判断它带什么值,于是 `already applied` 悄悄保留了过期的 num_warps=4,harness
报出 2.6-2.9x —— 那个 2.3 倍虚高恰好等于基线被压的倍数。所以本脚本**把启动那
一行原文打出来**,再决定要不要改,并且改在副本上,不动机器上已有的接线。

**收尾会做一次反算校验。** 从 harness 自己的延迟反算 GB/s,和已知的 465 GB/s
(32768x64,上游默认)对一下。这条规矩也是那次事故留下的:相信任何加速比之前,
先反算带宽。
"""

import glob
import importlib.util
import inspect
import os
import re
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402
try:
    import torch_musa  # noqa: F401
except ImportError:
    pass
import flaggems_vllm  # noqa: E402

DEVFN = flaggems_vllm.runtime.torch_device_fn
CEIL = 1332.0


def find_baseline():
    """找机器上已接好的 vLLM XPU 基线,**用 AST,不执行任何东西**。

    上一版按字符串搜,然后直接 exec 第一个命中的文件 —— 而命中的是可运行脚本
    (包括这个探针自己的副本),exec 把它们整个跑了一遍,于是同一条错误打印了
    十六次。按内容搜出来的文件不能拿来 exec:它可能是接线脚本、可能有副作用、
    也可能就是本文件。

    改成先 parse,只有当文件在**顶层定义**了 `xpu_qnorm_rope*` 函数时才算候选,
    并且把匹配到的函数名一并返回,后面按名字取,不再靠 dir() 猜。
    """
    import ast as _ast
    out = []
    for f in sorted(set(glob.glob(os.path.join(REPO, "myowncode", "**", "*.py"),
                                  recursive=True))):
        if os.path.abspath(f) == os.path.abspath(__file__):
            continue
        try:
            tree = _ast.parse(open(f, errors="replace").read())
        except (OSError, SyntaxError):
            continue
        names = [n.name for n in tree.body
                 if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                 and n.name.startswith("xpu_qnorm_rope")]
        if names:
            out.append((f, names[0]))
    return out


def main():
    print("=" * 76)
    print("  S5000 — 基线用上游原样的启动配置(不传 num_warps)")
    print("=" * 76)
    print("  device:", DEVFN.get_device_name(0), " count:", DEVFN.device_count())

    cands = find_baseline()
    print("\n顶层定义了 xpu_qnorm_rope* 的文件(AST 判定,未执行任何文件):")
    for f, nm in cands:
        print("  {}   ->  {}()".format(f, nm))
    if not cands:
        print("  一个都没有。")
        print("  参考:含有 _xpu_qnorm_rope_kernel 字样但没有该函数定义的文件 ——")
        for f in sorted(glob.glob(os.path.join(REPO, "myowncode", "**", "*.py"),
                                  recursive=True)):
            try:
                if "_xpu_qnorm_rope_kernel" in open(f, errors="replace").read():
                    print("    " + f)
            except OSError:
                pass
        print("\n  这台机器上没有接好的完整 vLLM XPU 基线 —— memory 记着它是")
        print("  local-only 手工接的,不在仓库里,可能已经不在了。")
        print("\n[RESULT] NO_BASELINE_WIRED")
        return
    path, entry = cands[0]

    spec = importlib.util.spec_from_file_location("xpu_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, entry)

    src = inspect.getsource(fn)
    print("\n" + "=" * 76)
    print("  启动那一行的原文 —— 不看标志位,看代码")
    print("=" * 76)
    launch = re.search(r"_xpu_qnorm_rope_kernel\[[^\]]*\]\((?:.|\n)*?\n    \)", src)
    print(launch.group(0) if launch else "  (没匹配到启动调用,打印整段)\n" + src)

    warps = re.findall(r"num_warps\s*=\s*(\w+)", src)
    print("\n  该函数里出现的 num_warps:", warps or "无 —— 与上游一致")
    if warps:
        print("  存在 num_warps,而本次要的是上游原样。请先确认这是接线加的,")
        print("  再决定怎么处理 —— 本脚本不改机器上的文件。")
        print("\n[RESULT] NUM_WARPS_PRESENT")
        return
    print("  与上游一致,可以直接当作“原样”来测。")

    # ---- 接进 harness ----
    from benchmark import conftest as cf
    from benchmark import consts

    assert cf.Config is None
    cf.Config = cf.BenchConfig()
    cf.Config.mode = consts.BenchMode.KERNEL
    cf.Config.bench_level = consts.BenchLevel.CORE
    cf.Config.query = False

    from importlib import import_module
    bm = import_module(
        "benchmark.test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")
    cls = bm.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark

    real_make = cls.make_input

    def make_input_freeing(param):
        DEVFN.empty_cache()
        yield from real_make(param)

    cls.make_input = staticmethod(make_input_freeing)

    import benchmark.base as base
    base.pytest.fail = lambda *a, **k: None

    bench = cls()
    bench.torch_op = fn

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
    print("\n" + "=" * 76)
    print("  22 个形状")
    print("=" * 76)
    bench.run()

    # ---- 反算校验:相信比值之前先反算带宽 ----
    print("\n" + "=" * 76)
    print("  反算校验(32768 x 64)")
    print("=" * 76)
    hit = [r for r in rows if r[0] == 32768 and r[1] == 64]
    if hit:
        n, h, vend, gems = hit[0]
        moved = (2 * n * h * 512 * 2 + n * 512 * 2 + n * 584 + n * 64 * 4 + n * 16)
        gv = moved / (vend / 1000) / 1e9
        print("  基线 {:.4f} ms -> {:.1f} GB/s = 天花板的 {:.1f}%".format(
            vend, gv, 100 * gv / CEIL))
        print("  单 kernel 上 num_warps=4 实测 465 GB/s(34.9%)。")
        print("  完整基线还含 FP8 量化+插入,所以这里应当略低于 465,")
        print("  但**远高于**它就说明接的不是被默认值压住的那份 —— 数字别信。")
    else:
        print("  32768x64 没跑出来,校验跳过。")

    print("\n  加速比 = 上游原样的 vLLM XPU 基线 / FlagGems,--mode kernel --level core")
    print("  对照:基线跑在自己最优 num_warps=1 时是 1.20x,那才是 PR 采用的数。")
    print("\n[RESULT] AS_SHIPPED_TABLE_OK")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
