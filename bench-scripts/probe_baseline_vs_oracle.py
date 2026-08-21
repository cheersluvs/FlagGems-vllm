"""An eager torch / torch_npu composition of this operator, for use as a baseline.

There is no vendor kernel and no portable Triton implementation that builds on
this card, so the only honest comparison left is "what it costs to do the same
work with framework operators" -- which is what the fused kernel exists to
replace.

Two decisions keep the comparison fair rather than flattering:

  * where a vendor fused op computes exactly the right thing, use it.
    `npu_rms_norm` with a unit gamma is bit-identical to this operator's
    weightless RMSNorm (measured, max|diff| = 0), so the Q normalisation is one
    op here, not five. A baseline made deliberately naive would inflate the
    ratio.
  * the FP8 encode has no equivalent -- torch_npu cannot even cast to
    float8_e4m3fn on this card -- so it is integer arithmetic, mirroring the
    kernel's own encoder op for op. That is not a handicap invented for the
    baseline; it is the only way to produce these bytes here at all.

An eager op costs about 13-15 us on this card against roughly 450 us for a
Triton launch, so at small shapes this composition is competitive with the fused
kernel. That is a real property of the backend and belongs in the result, not
hidden by choosing shapes that avoid it.
"""

import torch
import torch_npu  # noqa: F401

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
HALF_ROPE_DIM = 32
QUANT_BLOCK = 64
NUM_QUANT_BLOCKS = 7
SCALE_BYTES_PER_TOKEN = 8
TOKEN_DATA_BYTES = 576
FP8_MAX = 448.0


def _rope_interleaved(x, cos, sin):
    """GPT-J RoPE over the last ROPE_DIM of x, pairs (2i, 2i+1).

    x: [..., ROPE_DIM] float32. cos/sin: [..., HALF_ROPE_DIM] float32.
    """
    pair = x.reshape(*x.shape[:-1], HALF_ROPE_DIM, 2)
    e = pair[..., 0]
    o = pair[..., 1]
    return torch.stack([e * cos - o * sin, e * sin + o * cos], dim=-1).reshape(
        *x.shape
    )


def _ue8m0_scale(block_max):
    """2**ceil(log2(block_max / FP8_MAX)) and its stored byte, from the bits.

    Not via exp2: on this card torch.exp2 is one ULP low for integer arguments,
    which would put every quantised value a ULP out and flip the ones sitting
    between two E4M3 codes.
    """
    raw = (block_max / FP8_MAX).to(torch.float32).contiguous()
    bits = raw.view(torch.int32)
    code = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    scale = (code << 23).contiguous().view(torch.float32)
    return scale, code.clamp(0, 255)


def _f32_to_e4m3_bits(x):
    """The kernel's encoder, op for op, in eager torch. Requires |x| <= 448."""
    b = x.contiguous().view(torch.int32)
    sign = (b >> 24) & 0x80
    mag = b & 0x7FFFFFFF
    e = (mag >> 23) - 120

    m_n = (mag >> 20) & 0x7
    round_n = (mag >> 19) & 1
    sticky_n = (mag & 0x7FFFF) != 0
    up = (round_n == 1) & (sticky_n | ((m_n & 1) == 1))
    m_n = m_n + up.to(torch.int32)
    ovf = m_n > 7
    e_n = e + ovf.to(torch.int32)
    m_n = torch.where(ovf, torch.zeros_like(m_n), m_n)

    # subnormal mantissa = round-to-nearest-even of |x| * 512, forced with 2^23
    magic = 8388608.0
    m_s = ((x.abs() * 512.0 + magic) - magic).to(torch.int32)

    v = torch.where(e >= 1, (e_n << 3) | m_n, m_s)
    return (sign | v).to(torch.uint8)


def eager_fused_deepseek_v4(
    q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, eps,
    cache_block_size,
):
    """Same contract as the operator: q is updated in place, k_cache written."""
    num_tokens, num_heads, _ = q.shape

    cs = cos_sin_cache[position_ids]                       # [n, ROPE_DIM]
    cos = cs[:, :HALF_ROPE_DIM]
    sin = cs[:, HALF_ROPE_DIM:]

    # ---- Q: weightless RMSNorm over HEAD_DIM, then RoPE on the last 64 -------
    # float32 in, float32 out, rounded to bfloat16 exactly once at the end --
    # the kernel's arithmetic. Feeding npu_rms_norm bfloat16 instead rounds the
    # normalised value before the rotation and again after, and that second
    # rounding put about 4% of q one step away from the kernel's answer.
    q2 = q.reshape(-1, HEAD_DIM).float()
    gamma = torch.ones(HEAD_DIM, dtype=torch.float32, device=q.device)
    qn = torch_npu.npu_rms_norm(q2, gamma, epsilon=eps)
    qn = qn[0] if isinstance(qn, (tuple, list)) else qn
    qn = qn.reshape(num_tokens, num_heads, HEAD_DIM)

    qc = cos[:, None, :].expand(num_tokens, num_heads, HALF_ROPE_DIM)
    qs = sin[:, None, :].expand(num_tokens, num_heads, HALF_ROPE_DIM)
    qn[..., NOPE_DIM:] = _rope_interleaved(qn[..., NOPE_DIM:], qc, qs)
    q.copy_(qn.to(q.dtype))

    # ---- KV: RoPE on the last 64, then UE8M0 FP8 quant + paged insert -------
    n_ins = slot_mapping.shape[0]
    kvf = kv[:n_ins].float()
    kv_rope = _rope_interleaved(
        kvf[:, NOPE_DIM:], cos[:n_ins], sin[:n_ins]
    ).to(kv.dtype)

    blk = kvf[:, :NOPE_DIM].reshape(n_ins, NUM_QUANT_BLOCKS, QUANT_BLOCK)
    block_max = blk.abs().amax(-1).clamp(min=1e-4)
    scale, code = _ue8m0_scale(block_max)
    x_uint8 = _f32_to_e4m3_bits(blk / scale[..., None]).reshape(n_ins, NOPE_DIM)

    valid = slot_mapping >= 0
    slot = slot_mapping.clamp(min=0)
    block_idx = slot // cache_block_size
    pos_in_block = slot % cache_block_size
    fp8_off = pos_in_block * TOKEN_DATA_BYTES
    scale_off = (cache_block_size * TOKEN_DATA_BYTES
                 + pos_in_block * SCALE_BYTES_PER_TOKEN)

    rows = block_idx[valid]
    fo = fp8_off[valid]
    so = scale_off[valid]
    k_cache[rows[:, None], fo[:, None]
            + torch.arange(NOPE_DIM, device=q.device)[None, :]] = x_uint8[valid]
    k_cache[rows[:, None], fo[:, None] + NOPE_DIM
            + torch.arange(ROPE_DIM * 2, device=q.device)[None, :]] = (
        kv_rope[valid].view(torch.uint8).reshape(-1, ROPE_DIM * 2)
    )
    k_cache[rows[:, None], so[:, None]
            + torch.arange(NUM_QUANT_BLOCKS, device=q.device)[None, :]] = (
        code[valid].to(torch.uint8)
    )
    k_cache[rows, so + NUM_QUANT_BLOCKS] = 0


# ------------------------------------------------- baseline against the ORACLE
import importlib.util
import sys
import traceback

REPO = "/home/secure/wuyuqing/workspace/FlagGems-vllm"
sys.path.insert(0, REPO)
sys.path.insert(0, REPO + "/src")

CACHE_BLOCK = 64


def load_oracle():
    """The test file's torch reference, imported directly.

    This is the independent check. Comparing the baseline to the OPERATOR only
    shows they agree; if both made the same mistake they would agree and both be
    wrong. The oracle was written separately, and it is what the suite judges
    the operator by.
    """
    import types
    pkg = types.ModuleType("tests")
    pkg.__path__ = [REPO + "/tests"]
    sys.modules.setdefault("tests", pkg)
    spec = importlib.util.spec_from_file_location(
        "tests.test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        REPO + "/tests/test_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make(n, h, neg=False):
    dev = "npu"
    nb = (n + CACHE_BLOCK - 1) // CACHE_BLOCK + 1
    bb = CACHE_BLOCK * (TOKEN_DATA_BYTES + SCALE_BYTES_PER_TOKEN)
    torch.manual_seed(0)
    q = torch.randn(n, h, HEAD_DIM, dtype=torch.float32).to(torch.bfloat16).npu()
    kv = torch.randn(n, HEAD_DIM, dtype=torch.float32).to(torch.bfloat16).npu()
    slot = torch.arange(n, dtype=torch.int64, device=dev)
    if neg and n > 3:
        slot[1] = -1
        slot[n // 2] = -1
    pos = torch.arange(n, dtype=torch.int64, device=dev)
    cs = torch.randn(max(4096, n), ROPE_DIM, dtype=torch.float32).npu()
    cache = torch.zeros(nb, bb, dtype=torch.uint8, device=dev)
    return q, kv, cache, slot, pos, cs


def main():
    t = load_oracle()
    print("oracle loaded from the test file\n")

    print("### the eager BASELINE against the test file's ORACLE\n")
    print("  {:<24} {:>14} {:>18} {:>12}".format(
        "shape", "q differs", "k_cache differs", "q rel<=1e-2"))
    for n, h, neg in ((17, 64, False), (64, 128, False), (1024, 64, False),
                      (64, 64, True)):
        q1, kv1, c1, sl, po, cs = make(n, h, neg)
        q2, kv2, c2 = q1.clone(), kv1.clone(), c1.clone()

        # oracle mutates q, kv and k_cache in place
        t.ref_impl(q1, kv1, c1, sl.clone(), po.clone(), cs.clone(), 1e-6,
                   CACHE_BLOCK)
        eager_fused_deepseek_v4(q2, kv2, c2, sl, po, cs, 1e-6, CACHE_BLOCK)
        torch.npu.synchronize()

        a, b = q1.cpu().float(), q2.cpu().float()
        dq = int((q1.cpu() != q2.cpu()).sum())
        dc = int((c1.cpu() != c2.cpu()).sum())
        close = torch.allclose(a, b, rtol=1e-2, atol=1e-2)
        print("  {:<24} {:>14} {:>18} {:>12}".format(
            "{}x{}{}".format(n, h, " (-1 slots)" if neg else ""), dq, dc,
            str(close)))
        del q1, q2, kv1, kv2, c1, c2
        torch.npu.empty_cache()

    print("\n### is npu_rms_norm exact at the operator's real widths?\n")
    torch.manual_seed(1)
    for n in (1024, 65536):
        x = torch.randn(n, HEAD_DIM, dtype=torch.float32).npu()
        g = torch.ones(HEAD_DIM, dtype=torch.float32, device="npu")
        ours = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-6)
        out = torch_npu.npu_rms_norm(x, g, epsilon=1e-6)
        got = out[0] if isinstance(out, (tuple, list)) else out
        d = (got.view(torch.int32) != ours.view(torch.int32))
        print("  {:>6} rows: {} of {} float32 words differ".format(
            n, int(d.sum()), d.numel()))
        del x, g, ours, got
        torch.npu.empty_cache()

    print("\n[RESULT] DONE")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
