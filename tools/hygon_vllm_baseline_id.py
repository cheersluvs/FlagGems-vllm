"""Who provides torch.ops._C.top_k_per_row_* on this box?

On MetaX the answer was mcoplib's C++ kernel, and knowing that changed how the
ratios were read. Hygon has a vllm_hcu platform plugin, so the op could come
from stock vLLM's CUDA source hipified for DCU, or from a vendor library that
registers over it. This prints, for both ops:

  - whether the op resolves, and its schema
  - which shared object registered it (torch.ops.loaded_libraries, plus a
    symbol scan of the vllm package as a fallback)
  - every dispatch key it has a kernel for, which is how a vendor override
    shows up (a CUDA key filled in by one library and a CompositeExplicit by
    another)
  - the vllm package's version and file location

Read-only.

    tools/vendor_probe.sh tools/hygon_vllm_baseline_id.py hygon_vllm_baseline_id
"""

import os
import subprocess
import sys
from importlib import import_module

import torch


def line(k, v):
    print(f"  {k:<28} {v}")


print("=== packages")
for name in ("vllm", "vllm_hcu"):
    try:
        m = import_module(name)
        line(name, f"{getattr(m, '__version__', '?')}  {getattr(m, '__file__', '?')}")
    except Exception as e:  # noqa: BLE001
        line(name, f"absent ({type(e).__name__})")

import vllm._custom_ops  # noqa: E402,F401  (this is what resolves torch.ops._C)

print("\n=== the ops")
for name in ("top_k_per_row_prefill", "top_k_per_row_decode"):
    have = hasattr(torch.ops._C, name)
    line(name, "resolved" if have else "ABSENT")
    if not have:
        continue
    try:
        packet = getattr(torch.ops._C, name)
        line("  schema", str(packet.default._schema))
        keys = torch._C._dispatch_dump(f"_C::{name}")
        kernels = [
            ln.strip()
            for ln in keys.splitlines()
            if ln.strip().startswith(
                ("CUDA", "CPU", "Composite", "Meta", "PrivateUse", "AutogradCUDA")
            )
        ]
        for k in kernels[:8]:
            line("  kernel", k[:110])
    except Exception as e:  # noqa: BLE001
        line("  introspection", f"{type(e).__name__}: {e}")

print("\n=== which library registered it")
libs = sorted(getattr(torch.ops, "loaded_libraries", []))
for lib in libs:
    line("loaded_library", lib)

so_candidates = []
for base in {os.path.dirname(import_module("vllm").__file__)} | {
    os.path.dirname(getattr(import_module(n), "__file__", "") or "/nonexistent")
    for n in ("vllm_hcu",)
    if n in sys.modules
}:
    for root, _, files in os.walk(base):
        for f in files:
            if f.endswith(".so"):
                so_candidates.append(os.path.join(root, f))
line("shared objects found", len(so_candidates))
for so in so_candidates[:40]:
    try:
        out = subprocess.run(
            ["nm", "-DC", so], capture_output=True, text=True, timeout=120
        ).stdout
    except Exception:  # noqa: BLE001
        continue
    hits = [ln for ln in out.splitlines() if "top_k_per_row" in ln]
    if hits:
        line("DEFINES top_k_per_row", os.path.basename(so))
        for h in hits[:6]:
            print(f"      {h.strip()[:150]}")
