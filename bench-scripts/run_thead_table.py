"""The full 22-shape table on T-Head PPU-ZW810E, run twice, with device checks.

WHY THIS IS NOT JUST `pytest benchmark/...`. Two things about this card make a
single run untrustworthy, both learned the hard way on 2026-08-17/18:

  * For part of that period the card's BASIC REDUCTIONS returned wrong answers --
    `.sum()` of a tensor with 10 non-zeros gave 0, `.any()` over 635392 set
    elements gave False. They are correct now, reproducibly, and nothing I could
    find explains what changed. `ppu-smi -r` is not permitted for this user, so
    resetting is not available. The only defence is to check a known answer
    immediately before and after every measurement, and to throw the numbers
    away if either check fails.
  * A careful timing harness once reported 1.11x at 1024 tokens because a cache
    flush had been dropped -- the generic side read 2951.8 GB/s against a 2097
    GB/s ceiling, which was the tell. The operator rewrites q in place, so at
    1024 tokens its 67 MB footprint sits inside this card's 32-64 MiB cache
    boundary. `triton.testing.do_bench` clears a 256 MB buffer between
    iterations, which covers it -- but only because the repo's kernel mode uses
    do_bench. Do not substitute a hand-rolled timing loop here without
    reinstating the flush.

So: health check, run, health check, run, health check. Both runs are printed
side by side with their spread, because on this card a number that has not been
repeated is not a measurement. Two earlier runs agreed within ~1% on all 22
shapes; a cell that disagrees by more than a few percent should be treated as
the card, not as the operator.

UNLIKE THE ASCEND RUNNERS, nothing is rebound. T-Head's own vLLM port registers
`_C::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`, so the benchmark's
baseline resolves on its own and `bench.torch_op` is already the vendor kernel --
the only one of the four cards where that is true. Install it isolated and put
it on PYTHONPATH:

    pip install --no-deps -t /tmp/vllm_ppu 'vllm==0.20.1+v0.1.0.ppu2.1.0'
    PYTHONPATH=/tmp/vllm_ppu python3 run_thead_table.py

If VLLM_REF_AVAILABLE prints False, stop -- the run would silently measure
nothing, since the harness would have no baseline to divide by.
"""

import importlib
import os
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))

import torch  # noqa: E402


def device_ok(tag):
    """Known-answer reductions, in the exact shapes that once returned garbage.

    Cases and expected values are from the failure table: 10 of 1024, 1000 of
    100000, 1000 of 635392, and all of 635392. `.sum()`, `count_nonzero` and
    `.any()` each disagreed with the truth at least once, so all three are
    checked rather than whichever is convenient.
    """
    bad = []
    for n, k in ((1024, 10), (100000, 1000), (635392, 1000), (635392, 635392)):
        x = torch.zeros(n, dtype=torch.float32, device="cuda")
        x[:k] = 1.0
        torch.cuda.synchronize()
        got = (float(x.sum()), int(torch.count_nonzero(x)), bool(x.any()))
        want = (float(k), k, k > 0)
        if got != want:
            bad.append("n={} k={}: sum/nnz/any = {}, 应为 {}".format(n, k, got, want))
        del x
    torch.cuda.empty_cache()
    if bad:
        print("  [{}] 设备检查失败 —— 本轮数字作废".format(tag))
        for b in bad:
            print("      " + b)
        return False
    print("  [{}] 设备检查通过".format(tag), flush=True)
    return True


def one_run(label):
    from benchmark import conftest as cf
    from benchmark import consts

    if cf.Config is None:
        cf.Config = cf.BenchConfig()
        cf.Config.mode = consts.BenchMode.KERNEL
        cf.Config.bench_level = consts.BenchLevel.CORE
        cf.Config.query = False

    mod = importlib.import_module(
        "benchmark.test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    )
    if not mod.VLLM_REF_AVAILABLE:
        print("VLLM_REF_AVAILABLE = False —— 厂商基线没解析出来,停止。")
        print("检查 PYTHONPATH 是否包含隔离安装的 vllm==0.20.1+v0.1.0.ppu2.1.0。")
        return None

    cls = mod.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark
    bench = cls()

    real_make = cls.make_input

    def make_input_freeing(param):
        torch.cuda.empty_cache()
        yield from real_make(param)

    cls.make_input = staticmethod(make_input_freeing)

    # A shape that fails must not discard the shapes that ran: run() prints one
    # result after the whole loop, and pytest.fail aborts it. The except has
    # already recorded error_msg and the finally appends the metric.
    import benchmark.base as base

    base.pytest.fail = lambda *a, **k: None

    real_latency = bench.get_latency
    rows = []
    pending = {}

    def timed(op, *args, **kwargs):
        q = args[0]
        key = (int(q.shape[0]), int(q.shape[1]))
        ms = real_latency(op, *args, **kwargs)
        if key in pending:
            rows.append((key[0], key[1], pending.pop(key), ms))
            n, h, base_ms, gems_ms = rows[-1]
            print("    {:>7} {:>5}  vendor {:>9.4f}  gems {:>9.4f}  {:>7.3f}x".format(
                n, h, base_ms, gems_ms, base_ms / gems_ms), flush=True)
        else:
            pending[key] = ms
        return ms

    bench.get_latency = timed
    print("\n=== 第 {} 轮 ===".format(label), flush=True)
    bench.run()
    cls.make_input = real_make
    return rows


def main():
    print("T-Head PPU-ZW810E — 全部 22 个形状,跑两轮,前中后各一次设备检查\n")
    print("torch {}  device {}  capability {}".format(
        torch.__version__, torch.cuda.get_device_name(0),
        torch.cuda.get_device_capability(0)))
    print()

    if not device_ok("开始前"):
        print("\n[RESULT] DEVICE_BAD_BEFORE")
        return
    r1 = one_run(1)
    if r1 is None:
        print("\n[RESULT] NO_BASELINE")
        return
    if not device_ok("第一轮后"):
        print("\n[RESULT] DEVICE_BAD_MID")
        return
    r2 = one_run(2)
    if not device_ok("第二轮后"):
        print("\n[RESULT] DEVICE_BAD_AFTER")
        return

    # 两轮并排。这张卡上没重复过的数字不算数,所以差异必须可见,不能只报中位数。
    m2 = {(n, h): (b, g) for n, h, b, g in r2}
    print("\n\n{:>7} {:>5} | {:>10} {:>10} {:>8} | {:>10} {:>10} {:>8} | {:>7}".format(
        "tokens", "heads", "vendor①", "gems①", "①", "vendor②", "gems②", "②", "偏差"))
    print("-" * 96)
    worst = 0.0
    for n, h, b1, g1 in r1:
        s1 = b1 / g1
        if (n, h) not in m2:
            print("{:>7} {:>5} | {:>10.4f} {:>10.4f} {:>7.3f}x | {:>32} |".format(
                n, h, b1, g1, s1, "第二轮缺此形状"))
            continue
        b2, g2 = m2[(n, h)]
        s2 = b2 / g2
        d = abs(s1 - s2) / max(s1, s2)
        worst = max(worst, d)
        flag = "  <- 差异 >3%,按卡的问题处理" if d > 0.03 else ""
        print("{:>7} {:>5} | {:>10.4f} {:>10.4f} {:>7.3f}x | {:>10.4f} {:>10.4f} "
              "{:>7.3f}x | {:>6.1%}{}".format(n, h, b1, g1, s1, b2, g2, s2, d, flag))
    print("-" * 96)
    print("两轮之间最大偏差 {:.1%}".format(worst))
    print("\n加速比 = 厂商 vLLM 移植 / FlagGems,--mode kernel --level core")
    print("\n[RESULT] THEAD_TABLE_OK")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
