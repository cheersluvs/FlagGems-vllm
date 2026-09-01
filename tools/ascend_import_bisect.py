#!/usr/bin/env python3
"""Which import poisons NPU initialisation?

On this 910B, `import torch, torch_npu` followed by `torch.randn(device="npu")`
works, and so does `torch.npu.get_device_properties(0)` on its own. But after
`import flaggems_vllm` every NPU call dies with

    SetPrecisionMode ... AclSetCompileopt(ACL_PRECISION_MODE) ... 500001
    plog: There is no valid so about OpsKernelInfoStore or GraphOptimizer

and the first sign of it is swallowed: device_info warns "fallback to
device_id=0" and "fallback to None", which read as a harmless degradation while
the process is in fact already unusable.

So bisect the import chain. Each candidate runs in its OWN subprocess, because
the failure is sticky -- once ACL is in the bad state nothing later in that
process is trustworthy, which is the same reason the preflight refuses to launch
two kernels in one run.

    source /usr/local/Ascend/cann/set_env.sh
    PYTHONPATH=src:$PYTHONPATH python tools/ascend_import_bisect.py

Prints, for each prefix of the chain, whether a device tensor can still be
allocated afterwards. The first FAIL names the culprit.
"""

import os
import subprocess
import sys

# In the order flaggems_vllm/__init__.py performs them, plus the pieces those
# pull in, so the first failure localises as tightly as possible.
STEPS = [
    "pass",
    "import triton",
    "import flaggems_vllm.runtime.backend.device_finder as _d",
    "from flaggems_vllm.runtime.backend import device_finder as _d; _d and None",
    "from flaggems_vllm import runtime",
    "from flaggems_vllm.utils import device_info as _di; _di.get_device_id()",
    "import flaggems_vllm.ops",
    "from flaggems_vllm import testing",
    "import flaggems_vllm",
]

PROBE = """
import torch
try:
    import torch_npu  # noqa: F401
except Exception as e:
    print("NO_TORCH_NPU", type(e).__name__)
    raise SystemExit(2)
a = torch.randn(1024, device="npu")
print("SUM", float(a.sum()))
"""


def main():
    print("=" * 78)
    print("  昇腾导入二分：哪一步之后 NPU 就分配不出张量了")
    print("=" * 78)
    print(f"  python  {sys.executable}")
    print(f"  ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH', '未设置')}")
    print(f"  {'导入步骤':<62}结果\n")
    env = dict(os.environ)
    first_fail = None
    for step in STEPS:
        code = step + "\n" + PROBE
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, timeout=600)
        if r.returncode == 0:
            verdict = "OK"
        else:
            tail = [ln for ln in r.stderr.strip().splitlines() if ln.strip()]
            # not truncated: see ascend_param_sweep for why
            verdict = "FAIL  " + (tail[-1] if tail else "?")
            if first_fail is None:
                first_fail = step
        label = step if len(step) <= 60 else step[:57] + "..."
        print(f"  {label:<62}{verdict}", flush=True)

    print()
    if first_fail is None:
        print("  全部通过 —— 说明毒化不在导入链里，而在更晚的调用中。")
    else:
        print(f"  第一个失败的步骤: {first_fail}")
        print("  它之前的都通过，所以问题就在这一步引入的代码里。")
        print("  注意 'pass' 那行必须 OK；它不 OK 就说明环境本身没准备好，")
        print("  与 flaggems_vllm 无关。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
