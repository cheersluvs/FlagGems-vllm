"""What does top_k_per_row have to work with on Hygon BW1000?

Answers, in one run, the questions that decided the shape of the MetaX work:

  1. which stack is this (vendor, device, warp size, SMs, smem, threads)
  2. is there a VENDOR baseline to be measured against
     (torch.ops._C.top_k_per_row_{prefill,decode}), or only torch.topk
  3. is TLE reachable at all -- module, bindings, and whether tle.gpu.alloc
     LOWERS (the MetaX lesson: the module importing proves nothing)
  4. does the generic operator RUN and agree with torch.topk, both ops
  5. a first, rough timing against whatever baseline exists

Nothing here writes to the repo; it is a read-only survey.

    PY=<python> tools/vendor_probe.sh tools/hygon_topk_probe.py hygon_topk_probe
"""

import sys
import traceback
from importlib import import_module

import torch


def line(k, v):
    print(f"  {k:<26} {v}")


def attempt(what, fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        line(what, f"FAILED {type(e).__name__}: {str(e).strip()[:110]}")
        return None


print("=== stack")
line("python", sys.executable)
line("torch", torch.__version__)
for mod in ("triton", "flag_gems", "flaggems_vllm", "vllm"):
    attempt(mod, lambda m=mod: line(m, getattr(import_module(m), "__version__", "?")))
import flaggems_vllm  # noqa: E402
from flaggems_vllm import runtime  # noqa: E402

line("vendor", getattr(runtime.device, "vendor_name", "?"))
line("device name", getattr(runtime.device, "name", "?"))
line("tle_enabled (descriptor)", getattr(runtime.device.info, "tle_enabled", None))
props = torch.cuda.get_device_properties(0)
line("device", getattr(props, "name", "?"))
for attr in (
    "multi_processor_count",
    "warp_size",
    "max_threads_per_block",
    "shared_memory_per_block",
    "total_memory",
):
    line(attr, getattr(props, attr, "?"))
attempt(
    "triton max_shared_mem",
    lambda: line(
        "triton max_shared_mem",
        __import__("triton").runtime.driver.active.utils.get_device_properties(0)[
            "max_shared_mem"
        ],
    ),
)

print("\n=== baseline: is there a vendor kernel for this op?")
for name in ("top_k_per_row_prefill", "top_k_per_row_decode"):
    have = hasattr(torch.ops._C, name)
    line(f"torch.ops._C.{name}", have)
attempt(
    "vllm._custom_ops",
    lambda: line(
        "vllm._custom_ops",
        f"{len([o for o in dir(import_module('vllm._custom_ops')) if not o.startswith('_')])} names",
    ),
)

print("\n=== TLE")
from flaggems_vllm.utils.triton_version_utils import has_triton_tle  # noqa: E402

line("has_triton_tle(3,6,0)", has_triton_tle(3, 6, 0))
tle = attempt("import tle", lambda: import_module("triton.experimental.tle.language"))
if tle is not None:
    line("import tle", "ok")
    from triton._C import libtriton  # noqa: E402

    for sym in (
        "make_swizzled_shared_encoding_attr",
        "create_local_pointers",
        "create_exclusive_cumsum",
        "get_memdesc_type",
    ):
        line(f"binding {sym}", hasattr(libtriton.ir.builder, sym))
    import triton  # noqa: E402
    import triton.language as tl  # noqa: E402

    @triton.jit
    def _k(out_ptr, N: tl.constexpr):
        buf = tle.gpu.alloc(
            [N],
            dtype=tl.int32,
            layout=None,
            scope=tle.gpu.smem,
            nv_mma_shared_layout=False,
        )
        lane = tl.arange(0, N)
        tl.store(tle.gpu.local_ptr(buf), lane * 2)
        tl.debug_barrier()
        tl.store(out_ptr + lane, tl.load(tle.gpu.local_ptr(buf)))

    def _run_tle():
        out = torch.zeros(256, dtype=torch.int32, device="cuda")
        _k[(1,)](out, N=256)
        torch.cuda.synchronize()
        ok = torch.equal(out, torch.arange(256, dtype=torch.int32, device="cuda") * 2)
        line(
            "tle.gpu.alloc lowers",
            f"{'yes, round-trip correct' if ok else 'compiles, WRONG values'}",
        )

    attempt("tle.gpu.alloc lowers", _run_tle)

print("\n=== does the generic operator run, and is it right?")
dec = import_module("flaggems_vllm.ops.top_k_per_row_decode")
pre = import_module("flaggems_vllm.ops.top_k_per_row_prefill")
line("HAS_TLE", f"decode={dec.HAS_TLE} prefill={pre.HAS_TLE}")
line("launch geometry", f"{dec._launch_geometry()} (warp, max threads)")

torch.manual_seed(0)
for B, V, K in ((1, 262144, 512), (8, 32768, 512), (64, 129280, 1024)):
    logits = torch.randn(B, V, dtype=torch.float32, device="cuda")
    want = torch.topk(logits, K, dim=1).values.sort(dim=1).values

    def _one(op):
        idx = torch.zeros(B, K, dtype=torch.int32, device="cuda")
        if op == "decode":
            lens = torch.full((B,), V, dtype=torch.int32, device="cuda")
            flaggems_vllm.top_k_per_row_decode(
                logits, 1, lens, idx, B, logits.stride(0), logits.stride(1), K
            )
        else:
            s = torch.zeros(B, dtype=torch.int32, device="cuda")
            e = torch.full((B,), V, dtype=torch.int32, device="cuda")
            flaggems_vllm.top_k_per_row_prefill(
                logits, s, e, idx, B, logits.stride(0), logits.stride(1), K
            )
        torch.cuda.synchronize()
        got = logits.gather(1, idx.long().clamp(0, V - 1)).sort(dim=1).values
        oob = int(((idx < 0) | (idx >= V)).sum())
        return (
            "OK" if (torch.allclose(got, want) and oob == 0) else f"WRONG (oob={oob})"
        )

    for op in ("decode", "prefill"):
        try:
            line(f"{op} B={B} V={V} K={K}", _one(op))
        except Exception as e:  # noqa: BLE001
            line(
                f"{op} B={B} V={V} K={K}",
                f"FAILED {type(e).__name__}: {str(e).strip()[:100]}",
            )
            traceback.print_exc(limit=3)

print("\n=== rough timing vs torch.topk (ms, 8 iters after 3 warmups)")


def timed(fn, iters=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


for B, V, K in ((1, 262144, 512), (64, 129280, 1024)):
    logits = torch.randn(B, V, dtype=torch.float32, device="cuda")
    lens = torch.full((B,), V, dtype=torch.int32, device="cuda")
    idx = torch.zeros(B, K, dtype=torch.int32, device="cuda")
    try:
        t_gems = timed(
            lambda: flaggems_vllm.top_k_per_row_decode(
                logits, 1, lens, idx, B, logits.stride(0), logits.stride(1), K
            )
        )
        t_torch = timed(lambda: torch.topk(logits, K, dim=1))
        line(
            f"decode B={B} V={V}",
            f"gems {t_gems:.3f}  torch.topk {t_torch:.3f}  "
            f"ratio {t_torch / t_gems:.2f}x",
        )
    except Exception as e:  # noqa: BLE001
        line(f"decode B={B} V={V}", f"FAILED {type(e).__name__}")
