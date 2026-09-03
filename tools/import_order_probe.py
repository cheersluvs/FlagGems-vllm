"""Which import order, if any, survives FlagTree 0.6.1's circular imports?

The cycle is inside the wheel, not in our code: triton.backends.ascend.testing
imports torch_npu at module scope, torch_npu pulls torch._inductor, and that
does `from triton.compiler.compiler import AttrsDescriptor` while
triton.backends is still being built -- so triton.spec.ascend.compiler's
`from ..backends import backends` finds a half-initialised module.

Reordering imports in our own probes cannot fix a cycle between two third-party
packages; the only question is whether some entry point breaks it.  Each
candidate runs in its own process, because a failed import leaves sys.modules
poisoned for everything after it.
"""

import os
import subprocess
import sys

ORDERS = {
    "triton_only": "import triton",
    "torch_then_triton": "import torch; import triton",
    "npu_then_triton": "import torch_npu; import triton",
    "triton_then_npu": "import triton; import torch_npu",
    "torch_npu_triton": "import torch; import torch_npu; import triton",
    "compiler_first": "import triton.compiler.compiler; import triton; import torch_npu",
    "backends_first": "import triton.backends; import triton; import torch_npu",
}
ENVS = {
    "autoload=0": {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
    "autoload=0,nocompile": {"TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
                             "TORCH_COMPILE_DISABLE": "1"},
    "default": {},
}

TAIL = ("; import triton.language as tl; print('OK', triton.__version__)")

print(f"{'env':<22} {'order':<20} result")
print("-" * 78)
for ename, extra in ENVS.items():
    for oname, stmt in ORDERS.items():
        env = dict(os.environ, **extra)
        env.pop("PYTHONWARNINGS", None)
        p = subprocess.run([sys.executable, "-c", stmt + TAIL],
                           capture_output=True, text=True, env=env, timeout=300)
        if p.returncode == 0:
            msg = p.stdout.strip().splitlines()[-1]
        else:
            last = [l for l in p.stderr.strip().splitlines() if l.strip()]
            msg = last[-1] if last else f"exit {p.returncode}"
            msg = msg.strip()
        print(f"{ename:<22} {oname:<20} {msg}")

print("\n--- and separately: is triton.experimental.tle importable at all?")
for stmt in ("import triton.experimental.tle",
             "import triton.experimental.tle.language as t; print(sorted(a for a in dir(t) if not a.startswith('_')))",
             "import triton.experimental.tle.language.gpu as g; print(sorted(a for a in dir(g) if not a.startswith('_')))"):
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD="0")
    p = subprocess.run([sys.executable, "-c", "import triton; " + stmt],
                       capture_output=True, text=True, env=env, timeout=300)
    out = p.stdout.strip() or (p.stderr.strip().splitlines() or ["?"])[-1]
    print(f"  {stmt[:60]:<62} -> {out[:120]}")
