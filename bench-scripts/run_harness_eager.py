"""The operator against an eager torch_npu composition, through the repo's harness.

WHAT THIS BASELINE IS, AND WHAT IT IS NOT. There is no vendor kernel for this
operator on 910B and vLLM's portable Triton fallback does not compile here, so
neither of the baselines the other cards in this PR use exists. What remains is
the honest question a fused kernel actually answers: what does it cost to do the
same work with framework operators? `eager_baseline.py` is that composition --
`npu_rms_norm` for the weightless RMSNorm, elementwise RoPE, integer FP8
encoding (torch_npu cannot cast to float8_e4m3fn at all), advanced-indexing
scatter into the paged cache.

It is a CONSTRUCTED baseline. It is not a vendor implementation, not upstream
code, and not what vLLM runs on this card -- vLLM runs nothing on this card. Any
number out of this must be labelled that way or it will be read as the same kind
of measurement as the C550 and BW1000 rows, which it is not.

Two things keep it from being a straw man, both decided before any number was
taken: where a vendor fused op computes the right thing it is used, so the Q
normalisation is one op rather than five; and it was validated against the test
file's own oracle with the operator removed from the middle, so the comparison
is not circular -- k_cache bit-identical on every shape checked, including the
mixed-negative-slot case.

THE BASELINE WILL NOT RUN THE LARGEST SHAPES, and that is left to happen. At
131072 x 128 the float32 copy of q alone is 34 GiB against 61 GiB of device
memory, with the bfloat16 original still live. Chunking it would be a hand
optimisation of the baseline -- exactly the thing that makes a comparison
flattering -- so the shapes it cannot reach are reported as unreachable rather
than made to fit. `pytest.fail` is neutered so one such shape does not discard
the rows that did run.

Mechanics as in run_harness_ab.py: the harness's `speedup` is
`latency_base / latency`, `latency_base` times `self.torch_op` and `latency`
times `self.gems_op`, so binding the eager composition into the empty torch_op
slot makes the harness's own SpeedUp column the number wanted -- computed by the
repo's code, at its shapes, in its `--mode kernel`.

Run:  REPO=/path/to/FlagGems-vllm python3 run_harness_eager.py
`eager_baseline.py` must sit beside this file.
"""

import importlib
import os
import sys
import traceback

REPO = os.environ.get("REPO", "/home/secure/wuyuqing/workspace/FlagGems-vllm")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))


def patch_randn(torch):
    """Build large bfloat16 tensors without an fp32 temporary.

    `torch.randn(dtype=bfloat16)` allocates one on this backend -- it asked for
    32 GiB to produce a 16 GiB tensor. `Tensor.normal_()` does too, which was a
    guess of mine the card disproved. Tile one small random block instead; the
    operator has no data dependent control flow, so repeated values do not
    change what is timed. This matters more here than in the other runners: the
    eager baseline needs every byte it can get.
    """
    real_randn = torch.randn
    cache = {}

    def randn_no_fp32_temp(*size, **kw):
        dtype = kw.get("dtype")
        dev = kw.get("device")
        if dtype is not torch.bfloat16 or dev is None:
            return real_randn(*size, **kw)
        shape = (
            size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
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


def agree(torch, mod, eager, gems):
    """The two must compute the same thing or the ratio compares nothing.

    k_cache is the quantised output -- one ULP there is a different byte -- so it
    must be exact. q is bfloat16 and the two reduce the variance over different
    shapes, so judge it by the repo's own tolerance, as the repo's own test does.
    """
    print("### does the eager composition compute what the operator computes?\n")
    ok = True
    for n, h in ((17, 64), (1024, 64), (64, 128)):
        p = mod.TestParam(
            num_tokens=n,
            num_heads=h,
            num_tokens_insert=n,
            block_size=64,
            max_pos=max(4096, n),
            eps=1e-6,
        )
        for inp in mod.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark.make_input(p):
            q, kv, kc, slot, pos, cs, eps, bs = inp
            q2, kc2 = q.clone(), kc.clone()
            eager(q, kv, kc, slot, pos, cs, eps, bs)
            gems(q2, kv, kc2, slot, pos, cs, eps, bs)
            torch.npu.synchronize()
            a, b = q.cpu().float(), q2.cpu().float()
            dc = int((kc.cpu() != kc2.cpu()).sum())
            close = torch.allclose(a, b, rtol=1e-2, atol=1e-2)
            ok = ok and dc == 0 and close
            print(
                "  {:<10} k_cache differing bytes {:>6}   q within rtol=1e-2 {}".format(
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

    # base.py binds Config by name at import, so this must come first.
    assert cf.Config is None, "conftest.Config was already configured"
    cf.Config = cf.BenchConfig()
    cf.Config.mode = consts.BenchMode.KERNEL
    cf.Config.bench_level = consts.BenchLevel.CORE
    cf.Config.query = False

    import torch
    import torch_npu  # noqa: F401

    patch_randn(torch)

    from eager_baseline import eager_fused_deepseek_v4

    mod = importlib.import_module(
        "benchmark.test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
    )
    cls = mod.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark
    bench = cls()
    gems = bench.gems_op

    print("=" * 78)
    print("  THE OPERATOR vs AN EAGER torch_npu COMPOSITION")
    print()
    print("  latency_base = eager_baseline.eager_fused_deepseek_v4")
    print("  latency      = the shipped Ascend override")
    print("  SpeedUp      = eager / fused")
    print()
    print("  This baseline is CONSTRUCTED, not a vendor kernel and not upstream")
    print("  code. vLLM has no implementation on this card ({}), and its"
          .format("torch.ops._C carries the op: {}".format(mod.VLLM_REF_AVAILABLE)))
    print("  portable Triton fallback does not compile here. So this answers")
    print("  'what do framework operators cost', not 'how much faster than vLLM'.")
    print()
    print("  Shapes the baseline cannot fit are reported as unreachable rather")
    print("  than chunked to make them fit.")
    print()
    print("  mode={}  level={}  metrics={}  warmup={}  iters={}".format(
        cf.Config.mode.value, cf.Config.bench_level.value, bench.metrics,
        cf.Config.warm_up, cf.Config.repetition))
    print("=" * 78)
    print()

    if not agree(torch, mod, eager_fused_deepseek_v4, gems):
        print("  They do not agree, so a ratio between them would be comparing")
        print("  two different computations.")
        print("\n[RESULT] BASELINE_DISAGREES")
        return
    print("  k_cache identical, q within the repo's own tolerance.\n")

    bench.torch_op = eager_fused_deepseek_v4

    real_make = cls.make_input

    def make_input_freeing(param):
        # The allocator keeps every shape's reservation, and this baseline needs
        # the room far more than the operator does.
        torch.npu.empty_cache()
        yield from real_make(param)

    cls.make_input = staticmethod(make_input_freeing)

    # A shape the baseline cannot fit must not discard the shapes that ran.
    # `run()` prints one result after the whole loop and `pytest.fail` aborts it;
    # the except has already recorded error_msg and the finally appends the
    # metric, so returning instead of raising continues the sweep.
    import benchmark.base as base

    base.pytest.fail = lambda *a, **k: None

    real_latency = bench.get_latency
    state = {"n": 0}

    def get_latency_verbose(op, *args, **kwargs):
        state["n"] += 1
        which = "eager" if state["n"] % 2 == 1 else "fused"
        q = args[0]
        try:
            ms = real_latency(op, *args, **kwargs)
        except RuntimeError as e:
            first = str(e).splitlines()[0][:70]
            print("    {} tokens={:>7} heads={:>4}  UNREACHABLE: {}".format(
                which, q.shape[0], q.shape[1], first), flush=True)
            torch.npu.empty_cache()
            raise
        print("    {} tokens={:>7} heads={:>4}  {:>10.4f} ms".format(
            which, q.shape[0], q.shape[1], ms), flush=True)
        return ms

    bench.get_latency = get_latency_verbose

    print("### timing, printed as it goes (the table follows at the end)\n")
    bench.run()
    print("\n  Reminder: SpeedUp above is eager torch_npu / this operator, on a")
    print("  card where vLLM provides nothing to compare against.")
    print("\n[RESULT] EAGER_HARNESS_OK")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] EAGER_HARNESS_FAILED")
sys.stdout.flush()
