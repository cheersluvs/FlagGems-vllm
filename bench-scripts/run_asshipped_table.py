"""任一后端的全 22 形状表:基线用上游原样的启动配置(不传 num_warps)。

对照的是 PR 里那张表 —— 基线跑在自己实测的最优 num_warps。两张表都要有,
PR 才能说清为什么选了最优点:一个启动参数就能抹平的差距不该算作贡献,
但那句话需要实测支撑,不能只是论证。

已测:S5000 大形状 as-shipped 2.76-2.92x,对照最优点 1.20x。

**「原样」的边界是启动配置。** 上游不传 num_warps(v0.27.1 与 main 的第 135 行
都没有,也无 @triton.autotune),取 Triton 默认 4。两个可移植性补丁必须留着,
否则 131072 那几格跑不完 —— 详见 build_vllm_xpu_baseline.py。

**两个常数按卡给,别跨卡沿用。**

  CEILING_GBS  该卡实测的拷贝天花板
  OPTIMUM_GBS  基线在自己最优 num_warps 下的已知带宽(32768x64)

后者用来做上界校验:as-shipped 必须**不高于**它。若接近或超过,说明接的不是
被默认值压住的那份 —— 这正是记录在案的那次事故:接线脚本的幂等检查只判断补丁
在不在、不判断带什么值,留下过期的 num_warps,harness 给出的数字本身没错,
错的是标签。反算带宽是能立刻抓住它的检查。

    CEILING_GBS=1340.3 OPTIMUM_GBS=604.8 REPO=$PWD python3 run_asshipped_table.py
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
# 按卡的实测常数。不设默认值 —— 沿用别的卡的天花板是最容易犯又最难发现的错。
_KNOWN = {            # vendor: (ceiling GB/s, baseline GB/s at its own optimum)
    "mthreads": (1332.0, 982.5),
    "hygon": (1340.3, 604.8),
}
_v = getattr(flaggems_vllm, "vendor_name", "?")
_d = _KNOWN.get(_v, (None, None))
CEIL = float(os.environ.get("CEILING_GBS") or (_d[0] or 0)) or None
OPT = float(os.environ.get("OPTIMUM_GBS") or (_d[1] or 0)) or None


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
    print("  基线用上游原样的启动配置(不传 num_warps)")
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
    if not hit:
        print("  32768x64 没跑出来,校验跳过。")
    elif CEIL is None:
        print("  这张卡({})没有记录的天花板,也没给 CEILING_GBS —— 无法校验。"
              .format(_v))
        print("  比值先别引用:反算不了带宽就没法确认接的是哪一份基线。")
    else:
        n, h, vend, gems = hit[0]
        moved = (2 * n * h * 512 * 2 + n * 512 * 2 + n * 584 + n * 64 * 4 + n * 16)
        gv = moved / (vend / 1000) / 1e9
        gg = moved / (gems / 1000) / 1e9
        print("  基线 {:.4f} ms -> {:>7.1f} GB/s = 天花板的 {:.1f}%".format(
            vend, gv, 100 * gv / CEIL))
        print("  gems {:.4f} ms -> {:>7.1f} GB/s = 天花板的 {:.1f}%".format(
            gems, gg, 100 * gg / CEIL))
        if OPT:
            print("\n  该基线在自己最优 num_warps 下的已知带宽:{:.1f} GB/s".format(OPT))
            if gv > OPT * 0.9:
                print("  **as-shipped 达到了最优点的 {:.0f}% —— 太高。**".format(
                    100 * gv / OPT))
                print("  说明接的很可能不是被默认值压住的那份,比值别引用。")
            else:
                print("  as-shipped 为最优点的 {:.0f}%,符合“被默认值压住”的预期。"
                      .format(100 * gv / OPT))

    print("\n  加速比 = 上游原样的 vLLM XPU 基线 / FlagGems,--mode kernel --level core")
    print("  对照:PR 采用的是基线跑在自己最优 num_warps 时的比值,不是这张表。")
    print("\n[RESULT] AS_SHIPPED_TABLE_OK")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
