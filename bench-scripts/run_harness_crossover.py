"""Where does HEAD stop being slower than c50ad93?

The A/B run through the harness found something its shape list hides: below 64
tokens HEAD is about 1.9x SLOWER than the commit it tuned, because HEAD splits
the work into two kernel launches (:466 and :481) where c50ad93 had one, and a
launch costs ~450 us on this card. Above 1024 tokens HEAD wins 6.6x/9.9x. The
benchmark's shapes jump straight from 64 to 1024, so the crossover has never
been measured -- only extrapolated (~190 tokens at 64 heads, ~95 at 128).

That extrapolation assumes `before` scales linearly through a region where it is
partly launch-bound itself, which is exactly the kind of assumption that has
been wrong on this card before. Measure it.

Same machinery as run_harness_ab.py -- the repo's Benchmark, its do_bench, its
kernel mode, old operator bound into the empty baseline slot -- with only the
shape list replaced. The agreement check is skipped: that run already proved the
two versions compute the same thing, and nothing here changes either of them.

Run from anywhere:  REPO=/path/to/FlagGems-vllm python3 run_harness_crossover.py
"""

import importlib
import importlib.util
import os
import subprocess
import sys
import traceback

REPO = os.environ.get("REPO", "/home/secure/wuyuqing/workspace/FlagGems-vllm")
OLD_REV = os.environ.get("OLD_REV", "c50ad93")
REL = (
    "src/flaggems_vllm/runtime/backend/_ascend/fused/"
    "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py"
)

# Dense through the predicted crossover, with 64 and 1024 kept as anchors so
# this run can be checked against the one that is already in hand.
TOKEN_COUNTS = [64, 96, 128, 160, 192, 256, 384, 512, 768, 1024]

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))


def load_old():
    src = subprocess.check_output(
        ["git", "-C", REPO, "show", "{}:{}".format(OLD_REV, REL)]
    ).decode()
    assert "flaggems_vllm" not in src, "the old file is not self-contained"
    path = "/tmp/ascend_op_{}.py".format(OLD_REV)
    with open(path, "w") as f:
        f.write(src)
    spec = importlib.util.spec_from_file_location("ascend_op_old", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert


def main():
    from benchmark import conftest as cf
    from benchmark import consts

    assert cf.Config is None, "conftest.Config was already configured"
    cf.Config = cf.BenchConfig()
    cf.Config.mode = consts.BenchMode.KERNEL
    cf.Config.bench_level = consts.BenchLevel.CORE
    cf.Config.query = False

    import torch
    import torch_npu  # noqa: F401

    mod = importlib.import_module(
        "benchmark.test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    )
    cls = mod.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark

    # Only the shapes change. Everything else is the harness as it ships.
    def crossover_params():
        return [
            mod.TestParam(
                n,
                h,
                num_tokens_insert=n,
                block_size=64,
                max_pos=4096,
                eps=1e-6,
            )
            for n in TOKEN_COUNTS
            for h in (64, 128)
        ]

    cls.get_performance_test_params = staticmethod(crossover_params)

    real_make = cls.make_input

    def make_input_freeing(param):
        torch.npu.empty_cache()
        yield from real_make(param)

    cls.make_input = staticmethod(make_input_freeing)

    old = load_old()
    bench = cls()
    bench.torch_op = old

    print("=" * 78)
    print("  CROSSOVER -- where HEAD overtakes c50ad93")
    print()
    print("  Torch Latency = {}, one kernel launch".format(OLD_REV))
    print("  Gems Latency  = HEAD, two kernel launches")
    print("  Speedup       = before / after. NOT a speedup over vLLM;")
    print("                  vLLM has no kernel on this card.")
    print()
    print("  tokens: {}".format(TOKEN_COUNTS))
    print("  mode={}  warmup={}  iters={}".format(
        cf.Config.mode.value, cf.Config.warm_up, cf.Config.repetition))
    print("=" * 78)
    print()

    # As in the A/B run: `run()` prints nothing until every shape is done, and
    # any failure calls pytest.fail and aborts the loop. Neither is wanted from
    # a script. The except has already recorded error_msg and the finally
    # appends the metric, so returning instead of raising costs nothing.
    import benchmark.base as base

    base.pytest.fail = lambda *a, **k: None

    real_latency = bench.get_latency
    seen = {"n": 0}

    def get_latency_verbose(op, *args, **kwargs):
        seen["n"] += 1
        ms = real_latency(op, *args, **kwargs)
        q = args[0]
        if seen["n"] % 2 == 1:
            get_latency_verbose.before = ms
            get_latency_verbose.shape = (q.shape[0], q.shape[1])
        else:
            n, h = get_latency_verbose.shape
            b = get_latency_verbose.before
            print(
                "    tokens={:>5} heads={:>4}   before {:>8.4f}   after {:>8.4f}"
                "   ratio {:>6.2f}x  {}".format(
                    n, h, b, ms, b / ms, "HEAD wins" if ms < b else ""
                ),
                flush=True,
            )
        return ms

    bench.get_latency = get_latency_verbose

    bench.run()
    print("\n  Reminder: ratio is c50ad93 / HEAD.")
    print("\n[RESULT] CROSSOVER_OK")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] CROSSOVER_FAILED")
sys.stdout.flush()
