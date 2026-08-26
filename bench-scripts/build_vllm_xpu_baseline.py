"""从上游重建 vLLM XPU 基线 —— 这台机器上原来手工接的那份已经不在了。

AST 判定确认:myowncode/ 下没有任何文件定义 xpu_qnorm_rope*,只有两个探针
提到过那个 kernel 名。memory 记着基线是 local-only 手工接的、不在仓库里 ——
它没了。

重建方式:从上游取两个文件、**校验 sha256**、抽出需要的两段、打两个可移植性
补丁、拼成一个自足模块。不装 vLLM(这台机器上没有),也不动仓库里的任何东西。

  上游 v0.27.1
    vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py
      sha256 77f77ddb4a489bd16f82bd43d8c50b375ea97b11fffa2c88bc6c1a1743a2a2e6
    vllm/models/deepseek_v4/common/ops/cache_utils.py
      sha256 922409cd4a2f23781b15a32332db946499d907ef876c954d251fe1da5db8e294

**为什么 cache_utils 只搬两个函数就够。** 整文件 1157 行、依赖一大堆
vllm.model_executor / vllm.platforms。但 AST 查过:36-227 这一段的自由名字只有
torch / triton / tl / current_platform / get_fp8_min_max,而后两个只出现在
200、202 行的 `if use_fnuz:` 分支里 —— 默认 use_fnuz=False,永不求值。

**两个补丁,以及为什么必须打。**

  1. int32 溢出(vllm#52416,无人接):`token_idx = tl.program_id(0)` 留在
     int32,q 超过 2^31 元素就回绕。benchmark 形状表到 131072x128,不打这个
     补丁大形状必死 —— 跑不完 22 格。
  2. 存储竞争(vllm#52415,PR #52582 已开未合并):KV 分支先无掩码全宽写一次
     未旋转的值,随后又往同样地址写旋转值,没有顺序约束。影响数值不影响计时,
     但既然是已知缺陷就一并修,免得有人拿这份基线去比对正确性。

**「上游原样」指的是启动配置,不是「完全未打补丁」。** 上游不传 num_warps
(v0.27.1 与 main 的第 135 行都没有,也无 @triton.autotune),取 Triton 默认 4。
这份重建保持那一点原样 —— 那正是本次要测的东西。
"""

import hashlib
import io
import os
import sys
import urllib.request

RAW = "https://raw.githubusercontent.com/vllm-project/vllm/v0.27.1/"
FILES = {
    "xpu": ("vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py",
            "77f77ddb4a489bd16f82bd43d8c50b375ea97b11fffa2c88bc6c1a1743a2a2e6"),
    "cache": ("vllm/models/deepseek_v4/common/ops/cache_utils.py",
              "922409cd4a2f23781b15a32332db946499d907ef876c954d251fe1da5db8e294"),
}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "vllm_xpu_baseline.py")


def fetch(key):
    path, want = FILES[key]
    body = urllib.request.urlopen(RAW + path, timeout=60).read()
    got = hashlib.sha256(body).hexdigest()
    print("  {:<6} {} sha256 {}".format(
        key, path.split("/")[-1], "一致" if got == want else "不一致!"))
    if got != want:
        print("    期望 " + want)
        print("    实得 " + got)
        raise SystemExit("上游文件与记录的不一致 —— 停止,不要在未知内容上打补丁")
    return body.decode()


print("=" * 74)
print("从上游取文件并校验")
print("=" * 74)
xpu = fetch("xpu").split("\n")
cache = fetch("cache").split("\n")

# cache_utils 36..227(1-based)= 索引 35..227
quant = "\n".join(cache[35:227]).rstrip()
# xpu 19..159 = 索引 18..159,即 kernel + 宿主函数
xpu_seg = "\n".join(xpu[18:159]).rstrip()

print()
print("=" * 74)
print("打补丁")
print("=" * 74)
n = 0
a = "    token_idx = tl.program_id(0)\n"
b = ("    # 补丁 1/2 —— vllm#52416:留在 int32 会在 q 超过 2^31 元素时回绕,\n"
     "    # benchmark 到 131072x128,不修跑不完。\n"
     "    token_idx = tl.program_id(0).to(tl.int64)\n")
assert a in xpu_seg, "int32 补丁锚点没命中"
xpu_seg = xpu_seg.replace(a, b, 1); n += 1
print("  1/2 int32 溢出:token_idx -> .to(tl.int64)")

a2 = "        tl.store(kv_out_base + offs, kv_full)"
b2 = ("        # 补丁 2/2 —— vllm#52415:这次全宽写会覆盖随后写入的旋转值,\n"
      "        # 两者无顺序约束。掩码方式照抄 Q 分支已有的 nope_mask。\n"
      "        tl.store(kv_out_base + offs, kv_full, mask=offs < NOPE_DIM)")
assert a2 in xpu_seg, "存储竞争补丁锚点没命中"
xpu_seg = xpu_seg.replace(a2, b2, 1); n += 1
print("  2/2 存储竞争:KV 批量 store 加 mask=offs < NOPE_DIM")

# 宿主函数里那句 from vllm...import 现在不需要了 —— 函数就在同一模块
old_imp = """    from vllm.models.deepseek_v4.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )
"""
if old_imp in xpu_seg:
    xpu_seg = xpu_seg.replace(
        old_imp, "    # (quantize_and_insert_k_cache 已在本模块内)\n", 1)
    print("  去掉宿主函数里对 vllm 的内部 import")

header = '''"""vLLM XPU 基线,从上游 v0.27.1 重建 —— 由 build_vllm_xpu_baseline.py 生成。

不要手改这个文件;改生成脚本。上游 sha256 与两个补丁的理由都在生成脚本的
docstring 里。**启动配置保持上游原样:不传 num_warps,取 Triton 默认 4。**
"""

import torch
import triton
import triton.language as tl

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HALF_ROPE = ROPE_DIM // 2

'''

io.open(OUT, "w", encoding="utf-8").write(
    header + quant + "\n\n\n" + xpu_seg + "\n")

import ast
src = io.open(OUT, encoding="utf-8").read()
ast.parse(src)
print()
print("=" * 74)
print("产物")
print("=" * 74)
print("  {}  ({} 行)".format(OUT, len(src.split("\n"))))
tree = ast.parse(src)
tops = [x.name for x in tree.body
        if isinstance(x, ast.FunctionDef)]
print("  顶层函数:", tops)
print("  补丁数:", n)
launch = [l for l in src.split("\n") if "_xpu_qnorm_rope_kernel[" in l]
print("  启动调用:", launch)
print("  该文件里的 num_warps:",
      [l.strip() for l in src.split("\n") if "num_warps" in l] or "无 —— 与上游一致")
print("\n[RESULT] BASELINE_BUILT")
