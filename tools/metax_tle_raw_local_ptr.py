"""Is the local_ptr shim still needed on a FlagTree that carries #1164?

The MetaX override does not hand the generic ops metax's own tle.gpu.local_ptr.
It wraps it so the returned pointer gains pid >> 31 -- always 0, and not
provably so: the prebuilt metaxTritonPlugin.so widens the vector width of an
unmasked shared load from pointer alignment alone, unclamped by elements per
thread, and asserts (LoadStoreOpToLLVM.cpp:427) below 4 elements per thread.
BLOCK_SIZE=512 on 8 warps is 1.

flagos-ai/FlagTree#1164 changed cmake/FlagTreeOptions.cmake and metax's
lib/Analysis/Alias.cpp and no plugin code, so in principle it cannot have
retired the shim. In practice the masked-atomic corruption once blamed on this
same plugin turned out to be the allocator that #1164 fixed, so this measures
rather than argues: run both operators, at both tiles, with the shim and with
metax's raw local_ptr.

Each variant runs in its own process with its own TRITON_CACHE_DIR. Both are
required: a kernel compiled by the other variant would be served from cache,
and the plugin's assert aborts the process instead of raising, which only a
child survives. A case that aborts is reported with its signal.

    PY=/data/wuyuqing/workspace/mctle-v2/bin/python \
        tools/vendor_probe.sh tools/metax_tle_raw_local_ptr.py metax_raw_ptr
    ... --pytest        also run the two operator test files under raw
    ... --case decode   one case only
"""

import argparse
import os
import subprocess
import sys
import tempfile
from importlib import import_module

# (tile, elements per thread at 8 warps) -- the assert's regime is < 4.
CASES = {
    "selftest": "the gate's own self-test kernel, 512 lanes on 8 warps",
    "decode": "decode, 56 rows, tile 512 (1 element/thread)",
    "prefill_narrow": "prefill (4100, 1025), tile 512 (1 element/thread)",
    "prefill_wide": "prefill (64, 129280), tile 1024 (2 elements/thread)",
}
VARIANTS = ("shim", "raw")
DETAILS = {}
ENV_RAW = "FLAGGEMS_METAX_RAW_LOCAL_PTR"


def load_gate(op):
    """The module holding ensure_tle for this operator. It is its own file in
    the three-file layout and the operator's own file in the two-file one."""
    base = "flaggems_vllm.runtime.backend._metax.fused"
    try:
        return import_module(f"{base}.top_k_per_row_tle")
    except ImportError:
        return import_module(f"{base}.top_k_per_row_{op}")


def use_raw_local_ptr(op):
    """Give the gate metax's own local_ptr back, before anything compiles."""
    import triton.experimental.tle.language as real_tle

    gate = load_gate(op)
    if gate._SHIM is None:
        return "this Triton has no TLE"
    gate._SHIM.gpu.local_ptr = real_tle.gpu.local_ptr
    return None


def pytest_configure(config):  # noqa: ARG001 - pytest plugin hook
    """Used as `pytest -p metax_tle_raw_local_ptr` with the env var set."""
    if os.environ.get(ENV_RAW) != "1":
        return
    for op in ("decode", "prefill"):
        why = use_raw_local_ptr(op)
        print(f"[raw local_ptr] {op}: {why or 'patched'}")


def decode_inputs(rows, dev):
    import torch

    torch.manual_seed(42)
    v, k = 262144, 512
    logits = torch.randn(rows, v, device=dev, dtype=torch.float32)
    seq = torch.full((rows,), v, dtype=torch.int32, device=dev)
    idx = torch.zeros((rows, k), dtype=torch.int32, device=dev)
    return (logits, 1, seq, idx, rows, v, 1, k)


def prefill_inputs(shape, dev):
    import torch

    rows, v, k, s0 = shape
    torch.manual_seed(42)
    buf = torch.randn((rows - 1) * s0 + v, device=dev, dtype=torch.float32)
    logits = torch.as_strided(buf, (rows, v), (s0, 1))
    starts = torch.zeros(rows, dtype=torch.int32, device=dev)
    ends = torch.full((rows,), v, dtype=torch.int32, device=dev)
    idx = torch.empty((rows, k), dtype=torch.int32, device=dev)
    return (logits, starts, ends, idx, rows, s0, 1, k)


def correct(args):
    import torch

    logits, idx, k = args[0], args[3], args[7]
    got = logits.gather(1, idx.long().clamp(0, logits.shape[1] - 1)).sort(dim=1).values
    want = torch.topk(logits, k, dim=1).values.sort(dim=1).values
    return bool(torch.equal(got, want))


def run_case(case, variant):
    """One case in this process. Prints RESULT <status>; the parent reads it."""
    import torch
    import triton

    import flaggems_vllm

    dev = flaggems_vllm.device
    op = "decode" if case in ("selftest", "decode") else "prefill"
    if variant == "raw":
        why = use_raw_local_ptr(op)
        if why:
            print(f"RESULT skipped {why}")
            return

    gate = load_gate(op)
    on = gate.ensure_tle(dev)
    if not on:
        print(f"DETAIL {gate.status()['why']}")
    if case == "selftest":
        print(f"RESULT {'on' if on else 'off'} {gate.status()['why']}")
        return
    if not on:
        print(f"RESULT off {gate.status()['why']}")
        return

    generic = import_module(f"flaggems_vllm.ops.top_k_per_row_{op}")
    if case == "decode":
        generic.NUM_THREADS_PER_BLOCK = 512
        generic.MULTIPLE_BLOCKS_PER_ROW_CONFIG = 4
        args = decode_inputs(56, dev)
    elif case == "prefill_narrow":
        generic.NUM_THREADS_PER_BLOCK = 512
        args = prefill_inputs((4100, 1025, 512, 1288), dev)
    else:
        generic.NUM_THREADS_PER_BLOCK = 1024
        args = prefill_inputs((64, 129280, 1024, 129280), dev)

    fn = getattr(generic, f"top_k_per_row_{op}")
    fn(*args)
    torch.cuda.synchronize()
    if not correct(args):
        print("RESULT wrong answers")
        return
    ms = triton.testing.do_bench(
        lambda: fn(*args), warmup=100, rep=400, return_mode="median"
    )
    print(f"RESULT ok {ms:.4f} ms")


def child(case, variant, root):
    cache = os.path.join(root, f"{variant}-{case}")
    env = dict(os.environ, TRITON_CACHE_DIR=cache)
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, __file__, "--child", case, "--variant", variant]
    p = subprocess.run(cmd, env=env, capture_output=True, text=True)
    detail = [
        ln[len("DETAIL ") :] for ln in p.stdout.splitlines() if ln.startswith("DETAIL ")
    ]
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            DETAILS[variant, case] = detail[0] if detail else ""
            return line[len("RESULT ") :], p.returncode
    if p.returncode < 0:
        return f"ABORTED signal {-p.returncode}", p.returncode
    tail = (p.stderr or p.stdout).strip().splitlines()
    return (tail[-1][:90] if tail else f"exit {p.returncode}"), p.returncode


def run_pytest(root):
    print("\n######## the two operator test files, raw local_ptr")
    env = dict(os.environ, TRITON_CACHE_DIR=os.path.join(root, "pytest-raw"))
    env[ENV_RAW] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        ["src", "tools", env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "metax_tle_raw_local_ptr",
        "tests/test_top_k_per_row_decode.py",
        "tests/test_top_k_per_row_prefill.py",
    ]
    p = subprocess.run(cmd, env=env, text=True)
    print(f"  pytest exit {p.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=sorted(CASES))
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--child", choices=sorted(CASES))
    ap.add_argument("--pytest", action="store_true")
    a = ap.parse_args()
    if a.child:
        run_case(a.child, a.variant)
        return 0

    cases = [a.case] if a.case else list(CASES)
    root = tempfile.mkdtemp(prefix="metax-raw-ptr-")
    print(f"triton caches under {root}; one per variant and case")
    print(f"{'case':<16} {'shim':<34} raw")
    verdict = []
    for case in cases:
        got = {}
        for variant in VARIANTS:
            got[variant], _ = child(case, variant, root)
        print(f"{case:<16} {got['shim'][:33]:<34} {got['raw'][:33]}")
        verdict.append((case, got))
    print()
    for case in cases:
        print(f"  {case:<16} {CASES[case]}")
    if any(DETAILS.values()):
        print("\nwhy a variant is off, untruncated:")
        for (variant, case), why in DETAILS.items():
            if why:
                print(f"  {variant}/{case}: {why}")

    def good(status):
        return status.startswith(("ok", "on"))

    shim_bad = [c for c, g in verdict if not good(g["shim"])]
    raw_bad = [c for c, g in verdict if not good(g["raw"])]
    if shim_bad:
        # Nothing can be concluded about raw while the shipped path is broken
        # here: an environment failure fails both variants identically.
        print(f"\ninconclusive: the shim variant itself failed {shim_bad}")
    elif raw_bad:
        print(f"\nshim still needed: with metax's own local_ptr, {raw_bad} fail")
    else:
        print("\nraw local_ptr ran every case: the shim may be removable")
    if a.pytest:
        run_pytest(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
