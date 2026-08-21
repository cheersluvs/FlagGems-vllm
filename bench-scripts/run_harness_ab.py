"""Before against after, measured by THIS REPO's benchmark harness.

`ab_optimisation.py` already answered this question with its own timing loop.
This answers it again through the harness, which matters for a different reason:
the number that goes in a PR should come from the machinery the PR is judged by,
using its shapes, its `do_bench` call and its `--mode kernel`, not from a script
written by the person reporting the number.

THE TRICK. `Benchmark.run()` computes `speedup = latency_base / latency`, where
`latency_base` times `self.torch_op` and `latency` times `self.gems_op`
(base.py:397-421). On Ascend `torch_op` is None -- no vLLM kernel exists here --
so that slot is empty. Bind the OLD operator into it and the harness's own
SpeedUp column becomes before/after, computed by code nobody adjusted for this.

READ THE OUTPUT CORRECTLY. The column says "SpeedUp" and it is NOT the PR's
speedup. It is c50ad93 divided by HEAD: what the tuning bought, against an
already-working Ascend kernel. There is no vLLM baseline on this card to divide
by. The banner below says so on every run so a pasted log cannot be misread.

Why the harness cannot be reached through pytest here: the benchmark file has
`@pytest.mark.skipif(not VLLM_REF_AVAILABLE, ...)`, evaluated when the decorator
is applied, so it is fixed at import and cannot be patched afterwards. Getting
past it would mean registering something under `torch.ops._C.<op>` -- pointing
the baseline at a fake. Driving `Benchmark` directly avoids inventing anything.

Run from anywhere:  REPO=/path/to/FlagGems-vllm python3 run_harness_ab.py
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

sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))


def load_old():
    """The operator as it stood at OLD_REV, imported beside the current one.

    The override file is self-contained -- torch, triton, tl and nothing from
    the package -- so both versions can live in one process. Assert that rather
    than assume it: if the old file ever imported `flaggems_vllm`, loading it
    here would quietly pull in the CURRENT package and time HEAD twice.
    """
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
    return mod.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert, len(src.splitlines())


def patch_randn(torch):
    """Build large bfloat16 tensors without an fp32 temporary.

    `torch.randn(dtype=bfloat16)` allocates one on this backend -- it asked for
    32 GiB to produce a 16 GiB tensor and ran out. `Tensor.normal_()` does too,
    which was a guess of mine the card disproved. So make one small random block
    and tile it. The operator has no data dependent control flow, so repeated
    values do not change what is timed.
    """
    real_randn = torch.randn
    cache = {}

    def randn_no_fp32_temp(*size, **kw):
        dtype = kw.get("dtype")
        dev = kw.get("device")
        if dtype is not torch.bfloat16 or dev is None:
            return real_randn(*size, **kw)
        shape = (
            size[0]
            if len(size) == 1 and isinstance(size[0], (tuple, list))
            else size
        )
        t = torch.empty(*shape, dtype=dtype, device=dev)
        n = t.numel()
        chunk = min(n, 1 << 22)
        key = (chunk, str(dev))
        if key not in cache:
            cache[key] = (
                real_randn(chunk, dtype=torch.float32).to(torch.bfloat16).to(dev)
            )
        seed = cache[key]
        flat = t.view(-1)
        for off in range(0, n, chunk):
            end = min(off + chunk, n)
            flat[off:end].copy_(seed[: end - off])
        return t

    torch.randn = randn_no_fp32_temp


def agree(torch, mod, old, new):
    """Prove the two versions still compute the same thing before timing them.

    A ratio between two different computations is not a speedup. Every
    optimisation step was checked bit-exact as it landed, so k_cache -- the
    quantised output, where one ULP is a different byte -- must be identical.
    q is bfloat16 and the two versions reduce the variance over differently
    shaped tiles, so its sum, and the rsqrt taken from it, can land one ULP
    apart; judge that by the repo's own tolerance rather than by bit equality.
    """
    print("### do the two versions still compute the same thing?\n")
    Param = mod.TestParam
    ok = True
    for n, h, ins in ((17, 64, 17), (1024, 64, 1024), (64, 128, 64)):
        p = Param(
            num_tokens=n,
            num_heads=h,
            num_tokens_insert=ins,
            block_size=64,
            max_pos=max(4096, n),
            eps=1e-6,
        )
        for inp in mod.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.make_input(p):
            q, kv, kc, slot, pos, cs, eps, bs = inp
            q2, kc2 = q.clone(), kc.clone()
            old(q, kv, kc, slot, pos, cs, eps, bs)
            new(q2, kv, kc2, slot, pos, cs, eps, bs)
            torch.npu.synchronize()
            a, b = q.cpu().float(), q2.cpu().float()
            dc = int((kc.cpu() != kc2.cpu()).sum())
            close = torch.allclose(a, b, rtol=1e-2, atol=1e-2)
            ok = ok and dc == 0 and close
            print(
                "  {:<12} k_cache differing bytes {:>6}   q within rtol=1e-2 {}".format(
                    "{}x{}".format(n, h), dc, close
                )
            )
            break
        torch.npu.empty_cache()
    print()
    return ok


def main():
    from benchmark import conftest as cf
    from benchmark import consts

    # base.py does `from .conftest import Config`, binding it by name, so this
    # has to happen before base.py is imported.
    assert cf.Config is None, "conftest.Config was already configured"
    cf.Config = cf.BenchConfig()
    cf.Config.mode = consts.BenchMode.KERNEL
    cf.Config.bench_level = consts.BenchLevel.CORE
    cf.Config.user_desired_metrics = None  # the default IS the three we want
    cf.Config.query = False

    import torch
    import torch_npu  # noqa: F401

    patch_randn(torch)

    mod = importlib.import_module(
        "benchmark.test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    )
    old, nlines = load_old()
    bench = mod.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark()
    new = bench.gems_op

    print("=" * 78)
    print("  BEFORE vs AFTER -- through the repo's own benchmark harness")
    print()
    print("  latency_base = {} ({} lines), the last enablement commit".format(
        OLD_REV, nlines))
    print("  latency      = HEAD, after tuning")
    print("  SpeedUp      = before / after.  THIS IS NOT A SPEEDUP OVER vLLM.")
    print("                 vLLM has no kernel on this card ({}),".format(
        "torch.ops._C carries the op: {}".format(mod.VLLM_REF_AVAILABLE)))
    print("                 so there is no external baseline to divide by here.")
    print()
    print("  mode={}  level={}  metrics={}  warmup={}  iters={}".format(
        cf.Config.mode.value, cf.Config.bench_level.value,
        bench.metrics, cf.Config.warm_up, cf.Config.repetition))
    print("  fp8 gate = {}".format(mod.is_support_fp8e4nv()))
    print("=" * 78)
    print()

    if not agree(torch, mod, old, new):
        print("  The two versions disagree beyond bfloat16 rounding, so a ratio")
        print("  between them would compare two different computations.")
        print("\n[RESULT] VERSIONS_DIFFER")
        return
    print("  k_cache identical, q within tolerance: same function, fair ratio.\n")

    # Bind the old operator into the empty baseline slot. Do it after `agree`,
    # so a failed check never reaches the timing path.
    bench.torch_op = old

    real_make = mod.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.make_input

    def make_input_freeing(param):
        # The allocator keeps every shape's reservation, so a later shape fails
        # on memory earlier ones no longer use.
        torch.npu.empty_cache()
        yield from real_make(param)

    mod.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.make_input = staticmethod(
        make_input_freeing
    )

    # ---- keep a late failure from throwing away every earlier row -----------
    #
    # `run()` builds one BenchmarkResult after the whole dtype loop and prints
    # THAT (base.py:443-450), so nothing reaches the terminal until all 22
    # shapes are done. Meanwhile any shape that raises calls `pytest.fail`
    # (base.py:391, 434), which aborts the loop. The sweep ends at 131072 tokens
    # x 128 heads -- 17 GiB for q alone -- so the largest shapes may well not
    # fit, and an OOM at shape 20 would discard the 19 that worked. On a box
    # reachable only by pasting, that is an expensive way to learn nothing.
    #
    # Two changes, neither touching what is timed:
    #
    #   * neuter `pytest.fail`. The `except` has already stored `error_msg` and
    #     the `finally` appends the metric either way, so returning instead of
    #     raising turns a fatal shape into a recorded failure and the sweep
    #     continues. No pytest session is running here for it to report to.
    #   * print each measurement as it is taken, so a hard device fault -- which
    #     this card does produce -- still leaves the completed rows on screen.
    import benchmark.base as base

    base.pytest.fail = lambda *a, **k: None

    real_latency = bench.get_latency
    which = {"n": 0}

    def get_latency_verbose(op, *args, **kwargs):
        which["n"] += 1
        tag = "before" if (which["n"] % 2) == 1 else "after "
        ms = real_latency(op, *args, **kwargs)
        q = args[0]
        print(
            "    {} tokens={:>7} heads={:>4}  {:>10.4f} ms".format(
                tag, q.shape[0], q.shape[1], ms
            ),
            flush=True,
        )
        return ms

    bench.get_latency = get_latency_verbose

    print("### timing, printed as it goes (the table follows at the end)\n")
    bench.run()
    print("\n  Reminder: the SpeedUp column above is c50ad93 / HEAD.")
    print("\n[RESULT] HARNESS_AB_OK")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] HARNESS_AB_FAILED")
sys.stdout.flush()
