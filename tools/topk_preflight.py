#!/usr/bin/env python3
"""Vendor-neutral preflight for top_k_per_row_{prefill,decode}.

Answers, before any benchmark is trusted, every environment question that decides
whether a run on this card means anything. Replaces the MTT-only version so the
remaining vendors (MetaX, Hygon, T-Head, Ascend) share one tool.

Each probe is isolated, so a hard failure still leaves a useful log above it.

    PYTHONPATH=src:$PYTHONPATH python tools/topk_preflight.py

NOTE the `:$PYTHONPATH`. On Ascend, CANN's graph engine imports the Python
module `tbe` off PYTHONPATH during init, so `PYTHONPATH=src python ...` --
which REPLACES the variable rather than prepending to it -- makes every NPU
call die with `AclSetCompileopt(ACL_PRECISION_MODE) ... 500001`, whose real
cause (`No module named 'tbe'`) appears only in ~/ascend/log/debug/plog/.
Harmless on NVIDIA and MTT, which is why the shorter form looks fine.
    PYTHONPATH=src:$PYTHONPATH python tools/topk_preflight.py --run prefill   # one launch
    PYTHONPATH=src:$PYTHONPATH python tools/topk_preflight.py --run decode

Run the bare form FIRST. Kernel launches are opt-in and one per process, because
on some backends a failed launch poisons the context and every later result in
that process is worthless.
"""

import argparse
import hashlib
import os
import sys
import traceback

import torch

import flaggems_vllm

W = 78
DEV = flaggems_vllm.device
VENDOR = flaggems_vllm.vendor_name


def hdr(t):
    print("\n" + "=" * W + f"\n  {t}\n" + "=" * W, flush=True)


def row(k, v):
    print(f"  {k:<34} {v}", flush=True)


def check(label, fn):
    try:
        row(label, fn())
    except Exception as e:  # noqa: BLE001 - a probe failing IS the datapoint
        row(label, f"!! {type(e).__name__}: {e}")


def _dev_fn():
    return flaggems_vllm.runtime.torch_device_fn


def stage_env():
    hdr("1. INTERPRETER / FRAMEWORKS")
    row("python", sys.version.split()[0])
    row("vendor / device", f"{VENDOR}  /  {DEV}")
    check("torch", lambda: torch.__version__)
    check("device available", lambda: _dev_fn().is_available())
    check("device count", lambda: _dev_fn().device_count())

    def _triton():
        import triton

        return f"{triton.__version__}   @ {os.path.dirname(triton.__file__)}"

    check("triton / flagtree", _triton)

    for mod in ("torch_musa", "torch_maca", "torch_npu"):
        if mod in sys.modules or os.path.isdir(mod):
            check(mod, lambda m=mod: __import__(m).__version__)


def stage_mthreads_llc():
    """MTT only: which llc runs decides whether anything compiles correctly."""
    if VENDOR != "mthreads":
        return
    hdr("1b. llc RESOLUTION (Moore Threads only)")
    print(
        "  FlagTree ships a fixed llc (md5 cec9ff66...); the MUSA 4.3.5 system one\n"
        "  (7d0a51d8...) aborts in MTGPU ISel. Both self-report LLVM 14.0.0, so\n"
        "  only the md5 separates them.\n"
    )
    try:
        import triton

        b = os.path.join(
            os.path.dirname(triton.__file__), "backends", "mthreads", "bin", "llc"
        )
        if os.path.isfile(b):
            with open(b, "rb") as f:
                m = hashlib.md5(f.read()).hexdigest()
            row("bundled llc md5", f"{m}  {'GOOD' if m.startswith('cec9ff66') else '??'}")
        else:
            row("bundled llc", "MISSING -> falls back to the broken system llc")
    except Exception as e:  # noqa: BLE001
        row("bundled llc", f"!! {e}")


def stage_tle():
    hdr("2. TLE AVAILABILITY -- decides which code path each op takes")
    print(
        "  has_triton_tle() only checks the IMPORT resolves. It does NOT check the\n"
        "  backend can lower tle.gpu.alloc / local_ptr to shared memory. If the\n"
        "  import works but lowering does not, the op takes the TLE path and fails\n"
        "  where the non-TLE fallback would have worked.\n"
    )

    def _gate():
        from flaggems_vllm.utils.triton_version_utils import has_triton_tle

        return f"has_triton_tle(3,6,0) = {has_triton_tle(3, 6, 0)}"

    check("version+import gate", _gate)

    def _vendor():
        from flaggems_vllm.ops.top_k_per_row_prefill import _vendor_tle_enabled

        return (
            f"VendorDescriptor.tle_enabled = "
            f"{getattr(flaggems_vllm.runtime.device.info, 'tle_enabled', '?')}"
            f"   -> effective {_vendor_tle_enabled()}"
        )

    check("vendor declares TLE", _vendor)

    def _syms():
        import triton.experimental.tle.language as tle

        out = []
        for path in ("gpu.alloc", "gpu.local_ptr", "gpu.smem", "cumsum"):
            obj = tle
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            out.append(f"{path}={'yes' if obj is not None else 'NO'}")
        return "  ".join(out)

    check("tle symbols", _syms)

    for op in ("top_k_per_row_prefill", "top_k_per_row_decode"):

        def _branch(op=op):
            g = getattr(flaggems_vllm, op).__globals__
            on = g.get("HAS_TLE")
            return f"HAS_TLE={on} -> {'TLE (smem)' if on else 'non-TLE (global)'}"

        check(f"{op} branch", _branch)


def stage_device():
    hdr("3. DEVICE GEOMETRY vs WHAT THE OP NEEDS")

    def _props():
        p = _dev_fn().get_device_properties(0)
        avail = [a for a in dir(p) if not a.startswith("_")]
        bits = []
        for a in (
            "name", "multi_processor_count", "warp_size",
            "max_threads_per_block", "shared_memory_per_block",
            "regs_per_multiprocessor", "major", "minor",
        ):
            if a in avail:
                bits.append(f"{a}={getattr(p, a)}")
        return "\n" + "\n".join(f"      {b}" for b in bits)

    check("device properties", _props)

    def _inventory():
        """When get_device_properties throws, fall back to the vendor tool.

        On Ascend it raises `SetPrecisionMode ... error code 500001`, so the
        launch geometry below is a hardcoded fallback rather than anything read
        off the device -- worth knowing before trusting a num_warps.
        Also shows whether the device is shared, which decides whether small
        shapes can be timed at all.
        """
        import shutil
        import subprocess

        for cmd in (["npu-smi", "info"], ["mthreads-gmi"], ["nvidia-smi"]):
            if not shutil.which(cmd[0]):
                continue
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
            return "\n" + "\n".join(f"      {ln}" for ln in lines[:14])
        return "no vendor inventory tool on PATH"

    check("device inventory", _inventory)
    def _geom():
        from flaggems_vllm.ops.top_k_per_row_prefill import (
            _launch_geometry,
            _num_warps,
        )

        warp, maxt = _launch_geometry()
        return (
            f"warp={warp} max_threads/block={maxt}  ->  BLOCK 512 uses "
            f"{_num_warps(512)} warps = {_num_warps(512) * warp} threads"
        )

    check("resolved launch geometry", _geom)
    row("op smem need (TLE path)", "prefill ~21 KB / decode ~25 KB @ top_k=1024")
    row("", "top_k=2048 -> ~29 / ~33 KB. MetaX C550 has only 64 KB: check above.")

    def _gate():
        """The wide-block gate is not in every build.

        It lives in the MTT override on the shipping branch and in the generic
        operator on the working branch, so probing one of them unconditionally
        reported an ImportError that looked like a finding about the card. It is
        not: absence here means the generic operator simply never widens.
        """
        import importlib

        for mod, fn in (
            ("flaggems_vllm.ops.top_k_per_row_prefill", "_wide_block_max_rows"),
            ("flaggems_vllm.runtime.backend._mthreads.fused.top_k_per_row_prefill",
             "_wide_max_rows"),
        ):
            try:
                f = getattr(importlib.import_module(mod), fn)
            except (ImportError, AttributeError):
                continue
            return (f"{fn}() = {f()}  (from {mod.split('.')[-2]})  -> "
                    f"num_rows <= that uses the wide block")
        return "not present in this build -- the generic op never widens"

    print()
    check("wide-block gate", _gate)
    print(
        "\n  Gate is SM-derived AND requires a 32-lane warp: num_warps is\n"
        "  BLOCK_SIZE // 32, so on a 64-lane part BLOCK=1024 would ask for 2048\n"
        "  threads. A gate of 0 means widening is disabled on this card and its\n"
        "  own crossover has never been measured."
    )


def stage_binding():
    hdr("4. BINDING: generic or vendor override?")
    for op in ("top_k_per_row_prefill", "top_k_per_row_decode"):

        def _bound(op=op):
            import importlib

            top = getattr(flaggems_vllm, op)
            generic = getattr(importlib.import_module(f"flaggems_vllm.ops.{op}"), op)
            which = "OVERRIDE" if top is not generic else "generic"
            return f"{which:<9} from {top.__module__}"

        check(op, _bound)
    print("\n  Expect 'generic' for both: no vendor override ships for either op.")


def stage_vllm():
    hdr("5. vLLM BASELINE")
    print(
        "  Importing vLLM is NOT proof the op exists -- check the symbol itself,\n"
        "  AFTER the import, with hasattr. Never dir(): torch.ops._C is a lazy\n"
        "  namespace and dir() lists only what has already been resolved.\n"
    )

    def _v():
        import vllm._custom_ops  # noqa: F401

        p = hasattr(torch.ops._C, "top_k_per_row_prefill")
        d = hasattr(torch.ops._C, "top_k_per_row_decode")
        return f"prefill={p}  decode={d}"

    check("torch.ops._C has", _v)
    print("\n  Both False -> no baseline on this card; benchmarks measure nothing.")


def run_kernel(which):
    hdr(f"6. MINIMAL LAUNCH: {which}")
    dev = DEV
    torch.manual_seed(0)
    num_rows, vocab, top_k = 1, 20000, 1024
    row("shape", f"num_rows={num_rows} vocab={vocab} top_k={top_k}")

    logits = torch.randn((num_rows, vocab), dtype=torch.float32, device=dev)
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)

    if which == "prefill":
        starts = torch.zeros((num_rows,), dtype=torch.int32, device=dev)
        ends = torch.full((num_rows,), vocab, dtype=torch.int32, device=dev)
        flaggems_vllm.top_k_per_row_prefill(
            logits, starts, ends, indices, num_rows,
            logits.stride(0), logits.stride(1), top_k,
        )
    else:
        # next_n is a plain int, NOT a tensor: a tensor makes it pointer<int32>
        # and the kernel dies at compile time on `row_id // next_n`.
        seq_lens = torch.tensor([vocab], dtype=torch.int32, device=dev)
        flaggems_vllm.top_k_per_row_decode(
            logits, 1, seq_lens, indices, num_rows,
            logits.stride(0), logits.stride(1), top_k,
        )

    _dev_fn().synchronize()
    ref = torch.topk(logits[0, :vocab], top_k, largest=True, sorted=False).indices
    got = indices[0].to(torch.int64)
    same = torch.equal(
        torch.sort(logits[0][got]).values, torch.sort(logits[0][ref]).values
    )
    row("launch", "OK")
    row("selected values match torch", "YES" if same else "NO  <-- accuracy bug")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["prefill", "decode"], default=None)
    args = ap.parse_args()

    print("=" * W)
    print(f"  top_k_per_row preflight -- vendor={VENDOR}")
    print("=" * W)

    stage_env()
    stage_mthreads_llc()
    stage_tle()
    stage_device()
    stage_binding()
    stage_vllm()

    if args.run:
        try:
            run_kernel(args.run)
        except Exception:  # noqa: BLE001
            hdr(f"6. MINIMAL LAUNCH: {args.run} -- FAILED")
            # to STDOUT, and flushed: print_exc() defaults to stderr, which is
            # unbuffered while a piped stdout is not, so the traceback overtook
            # the report and landed above it -- where `| tail` hid the only part
            # that mattered.
            sys.stdout.flush()
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
            print("\n  Re-run the other op in a FRESH process; this context may be\n"
                  "  poisoned and later results from it cannot be trusted.\n")
            return 1
    else:
        hdr("NO KERNEL LAUNCHED")
        print("  Send this output back before running anything else. Then, in\n"
              "  separate processes:\n"
              "      python tools/topk_preflight.py --run prefill\n"
              "      python tools/topk_preflight.py --run decode\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
