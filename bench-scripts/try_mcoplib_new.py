"""Does a newer mcoplib make the MetaX C550 baseline launchable?

BACKGROUND. `mcoplib._C` registers
`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`, so the benchmark resolves a
baseline -- but on C550 the quant kernel FAILS TO LAUNCH at every shape. Root
cause verified from MetaX's own source: in `mcoplib-0.4.6`'s
`op/vllm/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` the quant launcher
calls `cudaLaunchKernelEx` with `config.attrs` left non-NULL while
`numAttrs == 0`, and `cudaLaunchConfig_t` is never zero-initialised.

**MetaX fixed it themselves**: 0.4.9 and 0.4.10 drop the Ex path entirely and use
an unconditional `<<<grid, kBlockSize, 0, stream>>>`. The installed wheel is
0.4.6. So the question this script answers is narrow: is a >= 0.4.9 wheel
published for this torch/MACA build, and does it actually launch here?

TWO TRAPS THIS SCRIPT IS BUILT AROUND.

1. **Do not infer the code path from the symbol table.** It is tempting to
   `nm -D` the .so and look for `cudaLaunchKernelEx`. That is not evidence: a
   from-source rebuild was observed lowering a plain `<<<...>>>` to
   `wcudaLaunchKernelExC` while 21 other launches in the same TU lowered to
   `mcLaunchKernel`. The MACA nvcc bridge chooses; the source form does not
   survive into the name. **The only ground truth is trying to launch.**

2. **A failed launch on this backend is asynchronous and lands somewhere else.**
   Left alone it surfaces on whatever call comes next. Forcing it takes a NEW
   kernel launch -- `synchronize()` alone is silent here, and so is a
   device-to-host copy. Hence the throwaway reduction after the call, which is
   the same technique `_skip_if_unrunnable` uses in the benchmark file.

NOTHING IS INSTALLED OVER THE WORKING ENVIRONMENT. A candidate wheel goes to
/tmp with `pip install --no-deps -t`, exactly as the T-Head vendor vLLM was
handled, and is reached through PYTHONPATH in a child process. The installed
0.4.6 is left untouched, so a failure here costs nothing.

    python3 try_mcoplib_new.py            # look, install to /tmp, test
    python3 try_mcoplib_new.py --lookonly # only report what is available
"""

import os
import re
import subprocess
import sys
import textwrap

TARGET = "/tmp/mcoplib_new"
OP = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
WANT = (0, 4, 9)          # first version whose source has no Ex path


def sh(cmd, timeout=600):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def parse(v):
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", v.strip())
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)


print("=" * 74)
print("1. 当前装的是什么")
print("=" * 74)
rc, out = sh("python3 -c \"import mcoplib, os; "
             "print('version', getattr(mcoplib,'__version__','?')); "
             "print('path', os.path.dirname(mcoplib.__file__))\"")
print(out.strip() or "(import mcoplib 失败)")
rc, out = sh("python3 -m pip show mcoplib 2>/dev/null | "
             "grep -E '^(Name|Version|Location)'")
print(out.strip() or "(pip show 无输出)")

print()
print("=" * 74)
print("2. 厂商源上有哪些版本")
print("=" * 74)
rc, out = sh("python3 -m pip config list 2>/dev/null")
print("pip 配置:\n" + textwrap.indent(out.strip() or "(空)", "  "))
rc, out = sh("python3 -m pip index versions mcoplib 2>&1")
print("\npip index versions:\n" + textwrap.indent(out.strip(), "  "))
avail = re.findall(r"(\d+\.\d+\.\d+)", out)
if not avail:
    # 老 pip 没有 index 子命令时的常用替代:故意请求一个不存在的版本,
    # 让错误信息把可用版本列出来。
    rc, out = sh("python3 -m pip install 'mcoplib==0.0.0.notexist' 2>&1 | tail -5")
    print("\n回退探测:\n" + textwrap.indent(out.strip(), "  "))
    avail = re.findall(r"(\d+\.\d+\.\d+)", out)

best = max((v for v in avail if parse(v) >= WANT), key=parse, default=None)
print("\n满足 >= 0.4.9 的最高版本: {}".format(best or "没有"))

if "--lookonly" in sys.argv or not best:
    if not best:
        print("\n厂商源上没有 >= 0.4.9 的 wheel。剩下的路是从源码重建:")
        print("  git clone --branch mcoplib-0.4.9 \\")
        print("      https://github.com/MetaX-MACA/mcoplib.git")
        print("  注意分支名 0.4.4 起从下划线改成了连字符 —— "
              "myowncode/build_metax_quant_baseline.py 里写的")
        print("  mcoplib_0.4.7 是个不存在的 ref。")
    print("\n[RESULT] LOOKED_ONLY")
    sys.exit(0)

print()
print("=" * 74)
print("3. 隔离安装到 {}(不动现有环境)".format(TARGET))
print("=" * 74)
rc, out = sh("rm -rf {t} && python3 -m pip install --no-deps -t {t} "
             "'mcoplib=={v}' 2>&1 | tail -6".format(t=TARGET, v=best))
print(out.strip())
if rc != 0:
    print("\n[RESULT] INSTALL_FAILED")
    sys.exit(1)
rc, out = sh("du -sh {}".format(TARGET))
print("体积: " + out.strip())

# ---- 子进程里试真启动 ----------------------------------------------------
child = r'''
import os, sys, traceback
sys.path.insert(0, os.environ["MCOPLIB_NEW"])
REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "src"))

import mcoplib
print("  子进程加载到的 mcoplib:", getattr(mcoplib, "__version__", "?"),
      os.path.dirname(mcoplib.__file__))
import mcoplib._C            # 注册必须在导入 benchmark 模块之前
import torch, flaggems_vllm
from importlib import import_module

mod = import_module("benchmark.test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")
print("  VLLM_REF_AVAILABLE =", mod.VLLM_REF_AVAILABLE)
if not mod.VLLM_REF_AVAILABLE:
    print("[RESULT] NOT_REGISTERED"); sys.exit(0)

# 直接取未经包装的算子:仓库那个 _skip_if_unrunnable 会把失败转成 skip,
# 而这里要的正是失败本身。
ref = getattr(torch.ops._C, "%s")
cls = mod.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark
p = mod.TestParam(64, 64, num_tokens_insert=64, block_size=64, max_pos=4096, eps=1e-6)
for inp in cls.make_input(p):
    q, kv, kc, slot, pos, cs, eps, bs = inp
    q2, kc2 = q.clone(), kc.clone()
    try:
        ref(q, kv, kc, slot, pos, cs, eps, bs)
        # 失败是异步上报的:单靠 synchronize 在这块后端是沉默的,
        # 必须再发一次真正的 kernel 才能把它逼出来。
        torch.zeros(1, device=flaggems_vllm.device).sum()
        torch.cuda.synchronize()
    except Exception as e:
        print("  启动失败:", str(e).splitlines()[0][:120])
        print("[RESULT] STILL_CANNOT_LAUNCH"); sys.exit(0)
    print("  启动成功")

    flaggems_vllm.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q2, kv, kc2, slot, pos, cs, eps, bs)
    torch.cuda.synchronize()
    dq = int((q.cpu() != q2.cpu()).sum())
    dc = int((kc.cpu() != kc2.cpu()).sum())
    print("  与 FlagGems 对比:q 差 {}, k_cache 差 {} 字节".format(dq, dc))
    break
print("[RESULT] LAUNCHES_OK")
''' % OP

path = "/tmp/_mcoplib_launch_test.py"
with open(path, "w") as f:
    f.write(child)

print()
print("=" * 74)
print("4. 真启动测试(独立进程,失败不会污染这里)")
print("=" * 74)
env = "MCOPLIB_NEW={} REPO={}".format(TARGET, os.environ.get("REPO", os.getcwd()))
rc, out = sh("{} python3 {} 2>&1 | tail -25".format(env, path), timeout=900)
print(out.strip())
print("\n[RESULT] DONE")
