"""Run THIS REPO's benchmark harness on Ascend, where there is no vLLM baseline.

The benchmark file cannot be invoked through pytest here: it carries

    @pytest.mark.skipif(not VLLM_REF_AVAILABLE, ...)

and that condition is evaluated when the decorator is applied, so the value is
fixed at import time and cannot be patched afterwards. The only way past it
through pytest would be to make `hasattr(torch.ops._C, OP_NAME)` true by
registering something under that name -- which would point `_VENDOR_REF` at a
fake and could produce an invented speedup. Not acceptable for a number that
goes in a PR.

But the harness itself does not need a baseline. `Benchmark.run()` only touches
`self.torch_op` when `latency_base` or `speedup` is among the requested metrics
(base.py:398-420), so `--metrics latency` exercises the operator alone. This
drives that path directly: same `Benchmark` subclass, same `get_latency`, same
shapes from the file's own `get_performance_test_params`, same mode and level --
everything except the pytest wrapper that the missing vLLM gates.

`Config` is a module global in benchmark/conftest.py that base.py binds by name
at import (`from .conftest import Config`), so it has to be set BEFORE the
benchmark module is imported.
"""

import os
import sys
import traceback

REPO = os.environ.get("REPO", "/home/secure/wuyuqing/workspace/FlagGems-vllm")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))


def main():
    from benchmark import conftest as cf
    from benchmark import consts

    # Must happen before base.py is imported, since it binds Config by name.
    assert cf.Config is None, "conftest.Config was already configured"
    cf.Config = cf.BenchConfig()
    cf.Config.mode = consts.BenchMode.KERNEL
    cf.Config.bench_level = consts.BenchLevel.CORE
    cf.Config.user_desired_metrics = ["latency"]
    cf.Config.query = False
    print("mode={} level={} metrics={} warmup={} iters={}".format(
        cf.Config.mode.value, cf.Config.bench_level.value,
        cf.Config.user_desired_metrics, cf.Config.warm_up, cf.Config.repetition))

    # Two pieces of measurement scaffolding, local to this run, neither of them
    # touching what is timed:
    #
    # 1. torch.randn(..., dtype=bfloat16) allocates an fp32 temporary on this
    #    backend -- it asked for 32 GiB to build a 16 GiB tensor and ran out.
    #    Building the tensor empty and filling it in place avoids that.
    # 2. The allocator keeps every shape's reservation, so a later shape fails
    #    on memory earlier ones no longer use. Release between shapes.
    import torch

    _real_randn = torch.randn
    _seed_cache = {}

    def randn_no_fp32_temp(*size, **kw):
        """Fill a large bf16 tensor without ever materialising it in fp32.

        Both `torch.randn(dtype=bfloat16)` and `Tensor.normal_()` allocate an
        fp32 temporary here -- each asked for 32 GiB to produce a 16 GiB tensor.
        (`normal_` doing so was a guess of mine that the card disproved.) So
        make one small random block and tile it. The operator has no data
        dependent control flow, so repeated values do not change what is timed.
        """
        dtype = kw.get("dtype")
        dev = kw.get("device")
        if dtype is not torch.bfloat16 or dev is None:
            return _real_randn(*size, **kw)
        shape = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) \
            else size
        t = torch.empty(*shape, dtype=dtype, device=dev)
        n = t.numel()
        chunk = min(n, 1 << 22)
        key = (chunk, str(dev))
        if key not in _seed_cache:
            _seed_cache[key] = (
                _real_randn(chunk, dtype=torch.float32).to(torch.bfloat16).to(dev)
            )
        seed = _seed_cache[key]
        flat = t.view(-1)
        for off in range(0, n, chunk):
            end = min(off + chunk, n)
            flat[off:end].copy_(seed[: end - off])
        return t

    torch.randn = randn_no_fp32_temp

    import importlib
    mod = importlib.import_module(
        "benchmark.test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    )
    print("VLLM_REF_AVAILABLE = {}   (so there is no baseline to divide by)"
          .format(mod.VLLM_REF_AVAILABLE))
    print("fp8 gate           = {}".format(mod.is_support_fp8e4nv()))

    bench = mod.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark()
    print("gems op            = {}".format(bench.gems_op.__module__))
    print("baseline           = {}\n".format(bench.torch_op))

    _real_make = mod.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.make_input

    def make_input_freeing(param):
        torch.npu.empty_cache()
        yield from _real_make(param)

    mod.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.make_input = staticmethod(
        make_input_freeing
    )

    bench.run()
    print("\n[RESULT] HARNESS_OK")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] HARNESS_FAILED")
sys.stdout.flush()
