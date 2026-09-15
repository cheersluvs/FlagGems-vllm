"""Re-measure the 910B row of PR #684 (operator vs eager torch_npu), one shape per process.

WHY A NEW DRIVER. `run_eager_isolated.py` reported eager FASTER than fused almost
everywhere (0.16x-0.67x) after the branch merged main. Nothing got faster: the
repo's harness now times Ascend kernel mode with
`triton.backends.ascend.testing.do_bench_npu` (benchmark/base.py:309), which
reports the MEAN DEVICE TIME PER KERNEL, not per callable. The eager composition
launches dozens of small kernels, so it is divided by dozens; the operator at
32768 x 64 issues 2 launches and read exactly half of its 9.92 ms. The August
table was taken with plain `triton.testing.do_bench` (eager 4.29 ms at 1 token,
not 3.3 us), which is also what every other vendor's row uses.

So each child measures the same shape with three timers, all through the
harness's own `Benchmark.get_latency` and `make_input`:

  npu     : kernel mode as the harness now does it (do_bench_npu, per kernel)
  kernel  : kernel mode with do_bench_npu replaced by triton.testing.do_bench,
            median -- exactly the non-Ascend branch; comparable to the PR table
  operator: operator mode, synchronised wall clock, host launch cost included

Nothing in the repository is edited; the replacement is a module attribute set
inside the child. eager_baseline.py is imported from EAGER_DIR (default
<repo>/myowncode). Shapes the eager side cannot fit are reported, not chunked.

    REPO=/path/to/FlagGems-vllm PYTHONPATH=/path/to/FlagGems-vllm/src:$PYTHONPATH \
        python3 bench-scripts/run_eager_rebench.py
"""

import importlib
import json
import os
import subprocess
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
EAGER_DIR = os.environ.get("EAGER_DIR", os.path.join(REPO, "myowncode"))
TOKENS = (1, 4, 17, 64, 1024, 2048, 8192, 32768, 65536, 98304, 131072)
HEADS = (64, 128)


def child(n, h):
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "src"))
    sys.path.insert(0, EAGER_DIR)
    from benchmark import conftest as cf
    from benchmark import consts

    cf.Config = cf.BenchConfig()
    cf.Config.mode = consts.BenchMode.KERNEL
    cf.Config.bench_level = consts.BenchLevel.CORE
    cf.Config.query = False

    import torch
    import triton
    import triton.backends.ascend.testing as ascend_testing
    import flaggems_vllm
    import eager_baseline

    base = importlib.import_module("benchmark.base")
    mod = importlib.import_module(
        "benchmark.test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")
    cls = mod.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark
    eager = getattr(eager_baseline, "eager_fused_deepseek_v4", None)
    if eager is None:
        cands = [k for k in dir(eager_baseline) if not k.startswith("_")]
        raise RuntimeError("eager_baseline has no eager_fused_deepseek_v4; has {}".format(cands))
    if os.environ.get("EAGER_CHUNK") == "1":
        # Whole below 65536x128 rows, token-chunked above; see eager_chunked.py
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import eager_chunked
        eager = eager_chunked.eager_chunked
    fused = flaggems_vllm.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
    fixed = hasattr(importlib.import_module(
        "flaggems_vllm.runtime.backend._ascend.fused"
        ".fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"), "launch_geometry")

    real_npu = ascend_testing.do_bench_npu

    def plain(fn, **_):
        return triton.testing.do_bench(fn, warmup=base.Config.warm_up,
                                       rep=base.Config.repetition, return_mode="median")

    bench = cls()
    p = mod.TestParam(n, h, num_tokens_insert=n, block_size=64, max_pos=4096, eps=1e-6)
    inp = next(iter(cls.make_input(p)))
    row = {"n": n, "h": h, "fixed_override": fixed,
           "eager_chunked": os.environ.get("EAGER_CHUNK") == "1"}

    def timed(label, op):
        try:
            return bench.get_latency(op, *inp)
        except Exception as e:
            lines = [x for x in str(e).splitlines() if x.strip()]
            row.setdefault("err", "{} raised: {}".format(label, (lines[0] if lines else type(e).__name__)[:70]))
            return None

    for timer in ("npu", "kernel", "operator"):
        base.Config.mode = (consts.BenchMode.OPERATOR if timer == "operator"
                            else consts.BenchMode.KERNEL)
        ascend_testing.do_bench_npu = real_npu if timer == "npu" else plain
        e1 = timed("eager", eager)
        f1 = timed("fused", fused)
        f2 = timed("fused", fused)
        e2 = timed("eager", eager)
        es = [x for x in (e1, e2) if x is not None]
        fs = [x for x in (f1, f2) if x is not None]
        row[timer] = [min(es) if es else None, min(fs) if fs else None]
        if "err" in row and not es:
            break
    ascend_testing.do_bench_npu = real_npu
    print("ROW " + json.dumps(row), flush=True)


def main():
    rows = []
    for n in TOKENS:
        for h in HEADS:
            log = "/tmp/eager_rebench_{}x{}.log".format(n, h)
            with open(log, "w") as f:
                p = subprocess.run([sys.executable, os.path.abspath(__file__), str(n), str(h)],
                                   stdout=f, stderr=subprocess.STDOUT, timeout=1800)
            r = None
            for line in open(log, errors="replace"):
                if line.startswith("ROW "):
                    r = json.loads(line[4:])
            if r is None:
                tail = [x.strip() for x in open(log, errors="replace").read().splitlines() if x.strip()][-1:]
                r = {"n": n, "h": h, "err": "child exit {}: {}".format(p.returncode, (tail[0] if tail else "")[:70])}
            rows.append(r)
            subprocess.run("pkill -f 'run_eager_rebench.py {} {}' 2>/dev/null".format(n, h), shell=True)
            print("  done {}x{}".format(n, h), flush=True)

    def ratio(pair):
        return "{:.2f}x".format(pair[0] / pair[1]) if pair and None not in pair else "-"

    def ms(v):
        return "{:.4f}".format(v) if v is not None else "-"

    print("\nPer-shape logs: /tmp/eager_rebench_<n>x<h>.log")
    print("fixed override installed: {}".format(all(r.get("fixed_override") for r in rows if "fixed_override" in r)))
    print("\n{:>7} {:>5} | {:>10} {:>10} {:>8} | {:>10} {:>10} {:>8} | {:>8}  note".format(
        "tokens", "heads", "eager ms", "fused ms", "kernel", "eager ms", "fused ms", "operator", "npu(bad)"))
    print("-" * 112)
    for r in rows:
        k, o, npu = r.get("kernel"), r.get("operator"), r.get("npu")
        print("{:>7} {:>5} | {:>10} {:>10} {:>8} | {:>10} {:>10} {:>8} | {:>8}  {}".format(
            r["n"], r["h"],
            ms(k[0]) if k else "-", ms(k[1]) if k else "-", ratio(k),
            ms(o[0]) if o else "-", ms(o[1]) if o else "-", ratio(o),
            ratio(npu), r.get("err", "")))
    print("\nkernel   = harness kernel mode with plain do_bench (median), as the PR table")
    print("operator = harness operator mode, host launch cost included")
    print("npu(bad) = harness kernel mode as-is: per-kernel means, not a valid ratio")
    print("\n[RESULT] EAGER_REBENCH_DONE")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        try:
            child(int(sys.argv[1]), int(sys.argv[2]))
        except Exception:
            traceback.print_exc()
        sys.stdout.flush()
    else:
        main()
