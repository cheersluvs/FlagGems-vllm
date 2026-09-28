"""ZW810E: the generic fp8 path against the removed integer encoder, A/B.

WHY NOW. FlagTree #1116 (merged 2026-09-24) declares fp8e4nv on PPU cap80-88 and
routes it through a software cast, so `.to(tl.float8e4nv)` -- which this card
rejected outright before -- now compiles. That is the precondition PR #684's own
text names for dropping the T-Head override ("once fp8e4m3fn is supported on
PPU, the generic path is the optimal"). The sentence predates the
implementation, so it is a hypothesis, not a result.

NO REBUILT FLAGTREE NEEDED. #1116 is on flagtree main only -- no release and no
0.7.0-rc branch contains it (rc2-triton3.6 was cut two hours before the merge).
It does not have to be installed: on the PPU the *downcast* it adds is pure
Python ("downcasts to f8e4m3nv are implemented in the frontend"), and this
operator only ever writes fp8. ppu_fp8_shim.py installs that same frontend code
as a monkey patch and verifies it byte-for-byte against torch before any timing
runs, so this box measures #1116's generic arm today. Only the fp8 -> float
upcast and the fp4/dot changes need the rebuilt compiler, and this operator uses
neither.

WHAT THE TWO ARMS ACTUALLY DIFFER IN. Both encode f32 -> OCP E4M3:

  generic   `.to(tl.float8e4nv)`, which on cap80 expands to
            `_downcast_f32_to_e4nv` in third_party/ppu/language/ppu/utils.py:
            ~30 elementwise ops with three PER-LANE variable shifts, two
            selects, a NaN test and satfinite clamping.
  override  `_f32_to_e4m3_bits` from bd8eb77^, magic-number RNE with no
            per-lane shift, no saturation branch (|x| <= 448 holds by
            construction: the scale is the smallest power of two with
            block_max / scale <= FP8_MAX) and no zero special case.

So the generic arm does strictly more work per element. Whether that is visible
is the question: at large shapes this operator ran at 96-98% of this card's copy
ceiling, where extra ALU can hide; the decode shapes are where it would show.

METHOD, from what this operator's earlier measurements cost:
  * correctness first -- if the two arms do not write the same k_cache bytes
    they are not the same function and a ratio between them is meaningless.
  * an A/A slot (generic timed twice) gives the noise floor; without it a few
    percent means nothing.
  * slot order rotates every round and every round is printed.
  * this card intermittently returns wrong answers from basic reductions, so a
    known-answer health check runs before and after each shape and its readings
    are discarded if either fails.

    REPO=/path/to/worktree PYTHONPATH=/tmp/vllm_ppu:$REPO/src:$PYTHONPATH \
        OVERRIDE_REV=bd8eb77^ python3 bench-scripts/probe_thead_fp8_ab.py
    THEAD_SHAPES=17x64,1024x64 ... to restrict the sweep
"""

import importlib
import importlib.util
import os
import statistics
import subprocess
import sys
import tempfile
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import triton  # noqa: E402

OP = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
REL = "src/flaggems_vllm/runtime/backend/_thead/fused/{}.py".format(OP)
OVERRIDE_REV = os.environ.get("OVERRIDE_REV", "bd8eb77^")
ROUNDS = int(os.environ.get("ROUNDS", 3))
SHAPES = [(1, 64), (1, 128), (4, 64), (4, 128), (17, 64), (17, 128), (64, 64), (64, 128),
          (1024, 64), (1024, 128), (2048, 64), (2048, 128), (8192, 64), (8192, 128),
          (32768, 64), (32768, 128), (65536, 64), (65536, 128), (98304, 64), (98304, 128),
          (131072, 64), (131072, 128)]
if os.environ.get("THEAD_SHAPES"):
    SHAPES = [tuple(int(v) for v in s.split("x")) for s in os.environ["THEAD_SHAPES"].split(",")]


def health(dev):
    """Known answers this card has been seen to get wrong. True == trustworthy."""
    a = torch.zeros(1 << 16, device=dev)
    a[::6554] = 1.0
    ok = int(a.sum()) == 10 and bool((a != 0).any())
    b = torch.arange(635392, device=dev) % 3 == 0
    return ok and int(b.sum()) == 211798


def load_override(dev):
    src = subprocess.run(["git", "-C", REPO, "show", "{}:{}".format(OVERRIDE_REV, REL)],
                         capture_output=True, text=True, check=True).stdout
    assert "_f32_to_e4m3_bits" in src, "that revision is not the integer-encoder override"
    p = os.path.join(tempfile.mkdtemp(), "thead_override.py")
    open(p, "w").write(src)
    spec = importlib.util.spec_from_file_location("thead_override", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, OP)


def _find_vendor_op():
    """The vendor baseline, from whichever vllm is importable.

    The PPU wheel `pip install -t /tmp/vllm_ppu` was not needed on this box: an
    installed vllm registers the op on import, so try that first and only report
    the arm missing if no namespace carries it (the timing then runs without it
    instead of aborting -- gen/ovr is the question this script exists for).
    """
    where = []
    try:
        import vllm
        where.append("vllm {} at {}".format(getattr(vllm, "__version__", "?"),
                                            os.path.dirname(vllm.__file__)))
        try:
            import vllm._C  # noqa: F401  (registers the custom ops)
        except Exception as e:
            where.append("vllm._C import failed: {}".format(str(e).splitlines()[0][:60]))
    except Exception as e:
        where.append("no importable vllm ({})".format(str(e).splitlines()[0][:60]))
    for ns in ("_C", "_C_cache_ops", "vllm", "_ppu_C"):
        lib = getattr(torch.ops, ns, None)
        op = getattr(lib, OP, None) if lib is not None else None
        if op is not None:
            return op, "torch.ops.{}.{} -- {}".format(ns, OP, "; ".join(where))
    return None, "not registered in torch.ops -- {}".format("; ".join(where))


def main():
    from benchmark import conftest as cf
    from benchmark import consts

    cf.Config = cf.BenchConfig()
    cf.Config.mode = consts.BenchMode.KERNEL
    cf.Config.bench_level = consts.BenchLevel.CORE
    cf.Config.query = False

    import flaggems_vllm
    import flaggems_vllm.ops as ops

    dev = flaggems_vllm.device
    fn = flaggems_vllm.runtime.torch_device_fn
    mod = importlib.import_module("benchmark.test_{}".format(OP))
    cls = getattr(mod, "FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark")

    # ---------------- preflight ----------------
    print("=" * 84)
    print("PREFLIGHT")
    print("=" * 84)
    print("  triton {}   vendor {}   capability {}".format(
        getattr(triton, "__version__", "?"), flaggems_vllm.vendor_name,
        __import__("flaggems_vllm.utils.device_info", fromlist=["x"]).get_device_capability()))
    try:
        u = importlib.import_module("triton.language.extra.ppu.utils")
        has_sw = hasattr(u, "convert_custom_float8_sub89")
    except Exception as e:
        has_sw = "import failed: {}".format(str(e).splitlines()[0][:50])
    print("  FlagTree #1116 software cast present: {}".format(has_sw))
    print("  FLAGTREE_LOW_PRECISION_FLOAT = {}".format(
        os.environ.get("FLAGTREE_LOW_PRECISION_FLOAT", "(unset -> default 1)")))
    import ppu_fp8_shim
    tdev = dev if isinstance(dev, torch.device) else torch.device(dev)
    if os.environ.get("THEAD_FP8_SHIM", "auto") == "0":
        got = {"mode": "native" if ppu_fp8_shim.native_cast_works(tdev)[0] else "unavailable",
               "reason": "THEAD_FP8_SHIM=0, shim not attempted"}
    else:
        got = ppu_fp8_shim.install(tdev)
    cast_mode = got["mode"]
    print("  f32 -> fp8e4nv cast: {}   ({})".format(
        {"native": "installed flagtree provides it", "shim": "via ppu_fp8_shim (#1116 frontend code)",
         "unavailable": "NOT AVAILABLE"}[cast_mode], got["reason"]))
    if cast_mode == "unavailable":
        print("\n  Without the cast the generic arm cannot run. The shim needs only the")
        print("  frontend, so this is a real incompatibility, not a missing build:")
        print("  read the reason above before rebuilding flagtree from main.")
        print("\n[RESULT] NO_FP8_CAST")
        return
    print("  health check: {}".format("pass" if health(dev) else "FAIL"))

    generic = getattr(ops, OP)
    override = load_override(dev)
    bound = getattr(flaggems_vllm, OP)
    print("  bound override is the generic function: {}".format(bound is generic))
    vendor, vendor_where = _find_vendor_op()
    print("  vendor vLLM kernel: {}".format(vendor_where))

    def inputs(n, h):
        p = mod.TestParam(n, h, num_tokens_insert=n, block_size=64, max_pos=4096, eps=1e-6)
        return next(iter(cls.make_input(p)))

    # ---------------- correctness: same function? ----------------
    print("\n" + "=" * 84)
    print("DO THE TWO ARMS WRITE THE SAME BYTES?  (a ratio between different")
    print("functions means nothing, so this gates the timing below)")
    print("=" * 84)
    same = True
    for n, h in ((17, 64), (1024, 64), (64, 128)):
        inp = inputs(n, h)
        q, kv, kc, slot, pos, cs, eps, bs = inp
        qa, ka = q.clone(), kc.clone()
        qb, kb = q.clone(), kc.clone()
        generic(qa, kv, ka, slot, pos, cs, eps, bs)
        override(qb, kv, kb, slot, pos, cs, eps, bs)
        fn.synchronize()
        dq = int((qa != qb).sum())
        dk = int((ka != kb).sum())
        same = same and dk == 0
        rel = 0.0
        if dq:
            a, b = qa.float(), qb.float()
            rel = float(((a - b).abs() / b.abs().clamp(min=1e-30))[qa != qb].max())
        print("  {:>7}x{:<4} k_cache differing bytes {:>8}   q differing {:>8}  max rel {:.2e}"
              .format(n, h, dk, dq, rel))
        del inp, qa, ka, qb, kb
        fn.empty_cache()
    print("  -> {}".format("same encoding" if same
                           else "DIFFERENT ENCODING; treat the timings as informational only"))

    # ---------------- timing ----------------
    print("\n" + "=" * 84)
    print("LATENCY, ms -- generic (software cast, {}) vs override (integer encoder)"
          .format(cast_mode))
    print("=" * 84)
    hdr = "  {:>7} {:>5} {:>10} {:>10} {:>10} {:>9} {:>9} {:>9}"
    print(hdr.format("tokens", "heads", "vendor", "generic", "override",
                     "gen/ovr", "vnd/gen", "vnd/ovr"))
    print("  " + "-" * 82)
    for n, h in SHAPES:
        if not health(dev):
            print("  {:>7} {:>5}   health check failed BEFORE the shape -- skipped".format(n, h))
            continue
        try:
            inp = inputs(n, h)
            q, kv, kc, slot, pos, cs, eps, bs = inp
            call = {}
            for name, impl in (("generic", generic), ("generic2", generic), ("override", override)):
                call[name] = (lambda impl=impl: impl(q.clone(), kv, kc.clone(), slot, pos, cs, eps, bs))
            if vendor is not None:
                call["vendor"] = lambda: vendor(q.clone(), kv, kc.clone(), slot, pos, cs, eps, bs)
            names = list(call)
            for k in names:
                call[k]()
            fn.synchronize()
            per = {k: [] for k in names}
            for r in range(ROUNDS):
                for k in names[r % len(names):] + names[:r % len(names)]:
                    per[k].append(triton.testing.do_bench(call[k], warmup=25, rep=300,
                                                          return_mode="median"))
            if not health(dev):
                print("  {:>7} {:>5}   health check failed AFTER the shape -- discarded".format(n, h))
                del inp
                fn.empty_cache()
                continue
            m = {k: statistics.median(v) for k, v in per.items()}
            g, o = m["generic"], m["override"]
            vnd = m.get("vendor")
            print(hdr.format(n, h,
                             "{:.4f}".format(vnd) if vnd else "—",
                             "{:.4f}".format(g), "{:.4f}".format(o),
                             "{:.3f}".format(g / o),
                             "{:.3f}".format(vnd / g) if vnd else "—",
                             "{:.3f}".format(vnd / o) if vnd else "—"))
            print("      A/A floor (generic twice) {:+.2%}   rounds gen {}  ovr {}".format(
                m["generic2"] / g - 1,
                " ".join("{:.4f}".format(x) for x in per["generic"]),
                " ".join("{:.4f}".format(x) for x in per["override"])))
            del inp
        except Exception as e:
            print("  {:>7} {:>5}   ERROR {}".format(n, h, str(e).splitlines()[0][:60]))
        fn.empty_cache()

    print("""
Reading it:
  gen/ovr within the A/A floor          -> the software cast costs nothing here;
                                           the generic path can replace the override
  gen/ovr above the floor at small
  shapes only                           -> as predicted: ALU-bound there, hidden by
                                           bandwidth at large shapes
  gen/ovr above the floor everywhere    -> the override earns its place; say so in
                                           the PR instead of the "generic is optimal"
                                           sentence, which was never measured
""")
    print("[RESULT] THEAD_FP8_AB_DONE")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
