"""上游原样启动:vLLM XPU 基线的 qnorm/rope kernel 在 flagtree 0.6.1 上会报错吗?

问题很窄:上游 **不传 num_warps**(v0.27.1 与 main 的第 135 行都是
`_xpu_qnorm_rope_kernel[grid](...)`,没有 @triton.autotune),所以它的“默认”
就是 Triton 的默认值 4。而老 llc 的 `Cannot select` 缺陷影响的是 num_warps
**1 和 2**,不是 4 —— 记录推出的结论是“默认一直编得过”,但那是从因果链推的,
没有在 0.6.1 上跑过。这个脚本去跑一次。

它同时把 1 / 2 / 4 显式扫一遍,这样“0.6.1 解锁了什么”是看得见的,而不是
只有一句结论。

**这里内嵌的 kernel 逐字取自上游 v0.27.1**
(`vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py`,
整文件 sha256 77f77ddb4a489bd16f82bd43d8c50b375ea97b11fffa2c88bc6c1a1743a2a2e6),
只把 `from vllm.triton_utils import tl, triton` 换成普通 triton 导入 —— 那台机器
没装 vLLM。**没有打我们的两个补丁**,这是有意的:

  * 存储竞争(vllm#52415,PR #52582 仍未合并)影响的是数值,不是能不能编;
  * int32 溢出(vllm#52416,无人接)在 32768x64 上不触发 ——
    (nt-1)*nh*512 = 1,073,709,056,小于 2^31。

所以在这个形状上,未打补丁的上游代码正好回答“能不能编、要多久”,
而**不能**用来判断结果对不对。

范围限定:只测 qnorm/rope 这一个 kernel,不是完整基线(完整基线还包含
FP8 量化+插入,走的是另一个 kernel)。
"""

import os
import sys
import time
import traceback

import torch
import triton
import triton.language as tl

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HALF_ROPE = ROPE_DIM // 2

@triton.jit
def _xpu_qnorm_rope_kernel(
    q_ptr,  # [num_tokens, num_heads, HEAD_DIM]
    kv_ptr,  # [num_tokens, HEAD_DIM]
    kv_out_ptr,  # [num_tokens, HEAD_DIM] bf16 (RoPE-applied kv for cache insert)
    position_ids_ptr,
    cos_sin_cache_ptr,
    eps: tl.constexpr,
    num_tokens,
    num_heads: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    """Apply per-head RMSNorm + GPT-J RoPE on Q, GPT-J RoPE on KV.

    GPT-J interleaved format: pairs are (data[2i], data[2i+1]).
    cos_sin_cache layout: [max_pos, ROPE_DIM] with first HALF_ROPE=cos,
    second HALF_ROPE=sin.
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if token_idx >= num_tokens:
        return

    pos = tl.load(position_ids_ptr + token_idx).to(tl.int64)

    # Load cos/sin for this position
    rope_pair_idx = tl.arange(0, HALF_ROPE)
    cos_val = tl.load(cos_sin_cache_ptr + pos * ROPE_DIM + rope_pair_idx).to(tl.float32)
    sin_val = tl.load(
        cos_sin_cache_ptr + pos * ROPE_DIM + HALF_ROPE + rope_pair_idx
    ).to(tl.float32)

    if head_idx < num_heads:
        # ========== Q: per-head RMSNorm + GPT-J RoPE ==========
        q_base = q_ptr + token_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM

        # Load full head
        offs = tl.arange(0, HEAD_DIM)
        q_vals = tl.load(q_base + offs).to(tl.float32)

        # RMSNorm (no weight)
        sq_sum = tl.sum(q_vals * q_vals, axis=0)
        rms = tl.rsqrt(sq_sum / HEAD_DIM + eps)
        q_vals = q_vals * rms

        # Store ONLY the NoPE portion (positions 0..NOPE_DIM-1)
        nope_mask = offs < NOPE_DIM
        tl.store(q_base + offs, q_vals.to(q_ptr.type.element_ty), mask=nope_mask)

        # GPT-J interleaved RoPE on the last ROPE_DIM dimensions:
        even_offs = NOPE_DIM + rope_pair_idx * 2
        odd_offs = NOPE_DIM + rope_pair_idx * 2 + 1

        # Re-load original values at rope positions and normalize
        q_even = tl.load(q_base + even_offs).to(tl.float32) * rms
        q_odd = tl.load(q_base + odd_offs).to(tl.float32) * rms

        new_even = q_even * cos_val - q_odd * sin_val
        new_odd = q_even * sin_val + q_odd * cos_val

        # Store rotated RoPE values
        tl.store(q_base + even_offs, new_even.to(q_ptr.type.element_ty))
        tl.store(q_base + odd_offs, new_odd.to(q_ptr.type.element_ty))
    else:
        # ========== KV: GPT-J RoPE only ==========
        kv_base = kv_ptr + token_idx * HEAD_DIM
        kv_out_base = kv_out_ptr + token_idx * HEAD_DIM

        # Copy full KV unchanged first
        offs = tl.arange(0, HEAD_DIM)
        kv_full = tl.load(kv_base + offs)
        tl.store(kv_out_base + offs, kv_full)

        # GPT-J interleaved RoPE on the last ROPE_DIM dimensions
        even_offs = NOPE_DIM + rope_pair_idx * 2
        odd_offs = NOPE_DIM + rope_pair_idx * 2 + 1

        kv_even = tl.load(kv_base + even_offs).to(tl.float32)
        kv_odd = tl.load(kv_base + odd_offs).to(tl.float32)

        new_even = kv_even * cos_val - kv_odd * sin_val
        new_odd = kv_even * sin_val + kv_odd * cos_val

        tl.store(kv_out_base + even_offs, new_even.to(kv_out_ptr.type.element_ty))
        tl.store(kv_out_base + odd_offs, new_odd.to(kv_out_ptr.type.element_ty))

CEIL = 1332.0  # S5000 实测天花板 GB/s(32768 tokens 处)


def env():
    print("=" * 74)
    print("环境")
    print("=" * 74)
    try:
        import importlib.metadata as md
        print("  flagtree/triton 版本:", md.version("flagtree"))
    except Exception:
        print("  flagtree 版本: 取不到")
    print("  triton.__version__:", getattr(triton, "__version__", "?"))
    # 版本字符串区分不了两个 llc(都自报 LLVM 14.0.0),所以比 md5
    llc = os.path.join(os.path.dirname(triton.__file__), "backends", "mthreads", "bin", "llc")
    if os.path.isfile(llc):
        import hashlib
        h = hashlib.md5(open(llc, "rb").read()).hexdigest()
        good = h == "cec9ff66714e311670b9412ec760e4aa"
        print("  自带 llc: 有  md5 {}  {}".format(h, "= 验证过的那个" if good else "!= 验证过的那个"))
    else:
        print("  自带 llc: 无 —— Triton 会回退到 MUSA 工具链的 llc")
        print("           (路径 {})".format(llc))
    print("  设备:", torch.cuda.get_device_name(0))


def make(n, h):
    torch.manual_seed(0)
    q = torch.randn(n, h, HEAD_DIM, dtype=torch.float32).to(torch.bfloat16).cuda()
    kv = torch.randn(n, HEAD_DIM, dtype=torch.float32).to(torch.bfloat16).cuda()
    kv_out = torch.empty_like(kv)
    pos = torch.arange(n, dtype=torch.int64, device="cuda")
    cs = torch.randn(max(4096, n), ROPE_DIM, dtype=torch.float32).cuda()
    return q, kv, kv_out, pos, cs


def run(n, h, warps):
    """warps=None 表示按上游原样启动 —— 一个 num_warps 都不传。"""
    q, kv, kv_out, pos, cs = make(n, h)
    grid = (n, h + 1)
    kw = {} if warps is None else {"num_warps": warps}

    def go():
        _xpu_qnorm_rope_kernel[grid](
            q, kv, kv_out, pos, cs, 1e-6, n,
            num_heads=h, HEAD_DIM=HEAD_DIM, ROPE_DIM=ROPE_DIM,
            NOPE_DIM=NOPE_DIM, HALF_ROPE=HALF_ROPE, **kw)

    try:
        go()
        torch.cuda.synchronize()
    except Exception as e:
        first = str(e).splitlines()[0][:100] if str(e) else type(e).__name__
        del q, kv, kv_out
        torch.cuda.empty_cache()
        return None, first

    for _ in range(5):
        go()
    torch.cuda.synchronize()
    best = None
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            go()
        torch.cuda.synchronize()
        t = (time.perf_counter() - t0) / 20
        best = t if best is None else min(best, t)
    del q, kv, kv_out
    torch.cuda.empty_cache()
    return best * 1e3, None


def main():
    env()
    n, h = 32768, 64
    # 这个 kernel 读 q 写 q,读 kv 写 kv_out,加 cos/sin 与 position
    moved = (2 * n * h * HEAD_DIM * 2 + 2 * n * HEAD_DIM * 2
             + n * ROPE_DIM * 4 + n * 8)
    print()
    print("=" * 74)
    print("32768 x 64,只测 qnorm/rope kernel")
    print("=" * 74)
    print("  {:<26} {:>10} {:>12} {:>9}".format("配置", "ms", "GB/s", "占天花板"))
    for label, w in (("上游原样(不传)", None), ("num_warps=1", 1),
                     ("num_warps=2", 2), ("num_warps=4", 4)):
        ms, err = run(n, h, w)
        if err:
            print("  {:<26} {:>10} {}".format(label, "失败", err))
        else:
            gbs = moved / (ms / 1000) / 1e9
            print("  {:<26} {:>10.4f} {:>12.1f} {:>8.1f}%".format(
                label, ms, gbs, 100 * gbs / CEIL))
    print()
    print("  上游 v0.27.1 与 main 都不传 num_warps,所以“原样”应与 num_warps=4 同值。")
    print("  天花板 1332 GB/s。数值正确性不在本脚本范围内 —— 未打存储竞争补丁。")
    print()
    print("[RESULT] UPSTREAM_DEFAULT_DONE")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
