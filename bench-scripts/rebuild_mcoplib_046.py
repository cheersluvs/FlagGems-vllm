"""Rebuild mcoplib 0.4.6's quant kernel with MetaX's own later launch fix.

WHY 0.4.6 AND NOT 0.4.9. 0.4.9 fixes the launch but is a different operator:
upstream vLLM broke this op's schema at v0.22.0 and mcoplib 0.4.9 follows it.

    0.4.6 / vLLM v0.21.0   (Tensor! q,   ..., float eps, int bs) -> ()
    0.4.9 / vLLM v0.22.0+  (Tensor q_in, ..., int q_head_padded,
                                             float eps, int bs) -> Tensor

`Tensor!` is the mutable marker: the old form rewrites q in place, the new one
takes it read-only and returns a fresh tensor. FlagGems-vllm implements the OLD
contract (tools/setup.sh pins VLLM_VERSION=0.20.2, and nothing in the tree
mentions q_head_padded), so 0.4.6 is the only version whose work is comparable.
Benchmarking against 0.4.9 would compare an in-place rewrite against an
out-of-place one -- same byte count, but writing to a cold buffer instead of
overwriting lines already resident from the read, which on this card's 32-64 MiB
cache is a structural difference at small shapes and nothing to do with kernel
quality.

THE FIX IS METAX'S OWN, AND ON THIS CARD IT CHANGES NOTHING SEMANTICALLY.
0.4.6 leaves `cudaLaunchConfig_t config;` uninitialised, sets `config.attrs`
unconditionally, then sets `config.numAttrs = (sm_version >= 90) ? 1 : 0` and
calls `cudaLaunchKernelEx`. Its own comment says "leave numAttrs = 0 and launch
as a regular kernel" -- the intent was already a plain launch. 0.4.9 drops the
Ex path entirely for `<<<grid, kBlockSize, 0, stream>>>`, which is what this
script does. C550 reports (8, 0), so numAttrs was 0 anyway and the attribute
being dropped is one that was never applied here. No feature is traded away.

IT IS STILL OUR PATCH ON VENDOR CODE. The result is not a vendor-published
wheel, and no such wheel exists to get: the box has no vendor index, the
/mnt/wheel it was installed from is gone, and every GitHub release on
MetaX-MACA/mcoplib carries assets=0. Any table using these numbers has to say
"vendor source, rebuilt with the vendor's own later fix", not "vendor kernel".

NO REGISTRATION COLLISION. The .cu carries no TORCH_LIBRARY or PYBIND11_MODULE
-- registration lives in op/vllm/torch_bindings.cpp -- so the rebuilt TU is
bound here under its own `mcoplib_rebuilt` namespace and the installed
mcoplib's torch.ops._C entries are untouched. The schema string below is copied
verbatim from 0.4.6's torch_bindings.cpp.

    REPO=/path/to/FlagGems-vllm python3 rebuild_mcoplib_046.py
"""

import os
import subprocess
import sys
import textwrap

WORK = "/tmp/mcoplib_rebuild"
SRC = os.path.join(WORK, "mcoplib")
REL = "op/vllm/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu"
REPO = os.environ.get("REPO", os.getcwd())


def sh(cmd, timeout=1800):
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       timeout=timeout)
    return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()


def head(n, t):
    print("\n" + "=" * 74 + "\n{}. {}\n".format(n, t) + "=" * 74)


head(1, "取 0.4.6 源码(tag,不是分支 —— mcoplib-0.4.6 分支不存在)")
if not os.path.isdir(SRC):
    rc, out = sh("mkdir -p {w} && git clone --depth 1 --single-branch "
                 "--branch mcoplib_0.4.6 "
                 "https://github.com/MetaX-MACA/mcoplib.git {s} 2>&1 | tail -4"
                 .format(w=WORK, s=SRC))
    print(out)
    if not os.path.isfile(os.path.join(SRC, REL)):
        print("\n[RESULT] CLONE_FAILED")
        sys.exit(1)
else:
    print("  已存在,复用 " + SRC)
cu = os.path.join(SRC, REL)
print("  " + cu)

head(2, "打补丁:去掉 cudaLaunchKernelEx 路径")
with open(cu) as f:
    text = f.read()

OLD = """  cudaLaunchConfig_t config;
  config.gridDim = dim3(grid);
  config.blockDim = dim3(kBlockSize);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = (sm_version >= 90) ? 1 : 0;

  cudaLaunchKernelEx(
      &config, fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel<scalar_t_in>,
      q_inout,"""

NEW = """  // PATCHED to match mcoplib 0.4.9, which drops the Ex path entirely.
  // 0.4.6 never zero-initialises `config` and sets `config.attrs` even when
  // numAttrs is 0, which is what fails the launch on C550. This card is
  // sm_80, so numAttrs was 0 regardless and the PDL attribute was never
  // applied -- the plain launch below is semantically identical here.
  fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel<scalar_t_in>
      <<<grid, kBlockSize, 0, stream>>>(
      q_inout,"""

if "PATCHED to match mcoplib 0.4.9" in text:
    print("  已打过补丁,跳过")
elif OLD in text:
    text = text.replace(OLD, NEW, 1)
    with open(cu, "w") as f:
        f.write(text)
    print("  已替换")
else:
    print("  找不到预期的代码块 —— 源码和记录的不一致,停止。")
    print("  期望看到的开头:\n" + textwrap.indent(OLD.splitlines()[0], "    "))
    print("\n[RESULT] PATCH_ANCHOR_MISSING")
    sys.exit(1)

rc, out = sh("cd {} && git diff --stat && git diff | head -40".format(SRC))
print(textwrap.indent(out, "  "))

head(3, "写绑定(独立命名空间,不碰 torch.ops._C)")
binding = os.path.join(WORK, "binding.cpp")
with open(binding, "w") as f:
    f.write('''#include <torch/library.h>
#include <torch/torch.h>

void fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    torch::Tensor& q, torch::Tensor const& kv, torch::Tensor& k_cache,
    torch::Tensor const& slot_mapping, torch::Tensor const& position_ids,
    torch::Tensor const& cos_sin_cache, double eps, int64_t cache_block_size);

// schema 逐字取自 0.4.6 的 op/vllm/torch_bindings.cpp
TORCH_LIBRARY(mcoplib_rebuilt, m) {
  m.def(
      "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert("
      "Tensor! q, Tensor kv, Tensor! k_cache, "
      "Tensor slot_mapping, Tensor position_ids, Tensor cos_sin_cache, "
      "float eps, int cache_block_size) -> ()");
}
TORCH_LIBRARY_IMPL(mcoplib_rebuilt, CUDA, m) {
  m.impl("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
         &fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert);
}
''')
print("  " + binding)

head(4, "编译并加载(完整日志,不截断)")
build = os.path.join(WORK, "build.py")
with open(build, "w") as f:
    f.write('''
import os, sys, torch
from torch.utils.cpp_extension import load
inc = os.path.join("%s", "op", "vllm")
# is_python_module=False 是必需的:绑定用的是 TORCH_LIBRARY,
# 它把算子注册进 torch 调度器,并不定义 PyInit_ 模块入口。
load(name="mcoplib_rebuilt",
     sources=["%s", "%s"],
     extra_include_paths=[inc],
     is_python_module=False,
     verbose=True)
print("BUILD_OK")
''' % (SRC, cu, binding))
log = os.path.join(WORK, "build.log")
rc, out = sh("cd {r} && python3 {b} > {l} 2>&1; echo rc=$?"
             .format(r=REPO, b=build, l=log))
with open(log, errors="replace") as f:
    blog = f.read()
if "BUILD_OK" in blog:
    print("  编译通过")
    print(textwrap.indent("\n".join(blog.strip().splitlines()[-6:]), "  "))
else:
    print("  编译失败,完整日志如下:\n")
    print(blog)
    print("\n[RESULT] BUILD_FAILED")
    sys.exit(1)

head(5, "真启动 + 对 FlagGems 验正确性(独立进程)")
test = os.path.join(WORK, "test.py")
with open(test, "w") as f:
    f.write('''
import os, sys
sys.path.insert(0, os.environ["REPO"]); sys.path.insert(0, os.path.join(os.environ["REPO"], "src"))
import torch
from torch.utils.cpp_extension import load
inc = os.path.join("%s", "op", "vllm")
load(name="mcoplib_rebuilt", sources=["%s", "%s"],
     extra_include_paths=[inc], is_python_module=False, verbose=False)
import flaggems_vllm
from importlib import import_module
mod = import_module("benchmark.test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")
cls = mod.FusedDeepseekV4QnormRopeKVRopeQuantInsertBenchmark
ref = torch.ops.mcoplib_rebuilt.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert

for n, h in ((64, 64), (1024, 64), (17, 128)):
    p = mod.TestParam(n, h, num_tokens_insert=n, block_size=64, max_pos=4096, eps=1e-6)
    for inp in cls.make_input(p):
        q, kv, kc, slot, pos, cs, eps, bs = inp
        q2, kc2 = q.clone(), kc.clone()
        try:
            ref(q, kv, kc, slot, pos, cs, eps, bs)
            # 失败是异步上报的:再发一次真 kernel 才能把它逼出来,
            # 单靠 synchronize 在这块后端是沉默的。
            torch.zeros(1, device=flaggems_vllm.device).sum()
            torch.cuda.synchronize()
        except Exception as e:
            print("  %%dx%%d 启动失败: %%s" %% (n, h, str(e).splitlines()[0][:110]))
            print("[RESULT] STILL_CANNOT_LAUNCH"); sys.exit(0)
        flaggems_vllm.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
            q2, kv, kc2, slot, pos, cs, eps, bs)
        torch.cuda.synchronize()
        dq = int((q.cpu() != q2.cpu()).sum())
        dc = int((kc.cpu() != kc2.cpu()).sum())
        print("  %%-10s 启动成功  q 差 %%d  k_cache 差 %%d 字节"
              %% ("%%dx%%d" %% (n, h), dq, dc))
        break
    torch.cuda.empty_cache()
print("[RESULT] REBUILD_WORKS")
''' % (SRC, cu, binding))
rc, out = sh("cd {r} && REPO={r} python3 {t} 2>&1 | tail -20"
             .format(r=REPO, t=test), timeout=1800)
print(out)
print("\n[RESULT] DONE")
