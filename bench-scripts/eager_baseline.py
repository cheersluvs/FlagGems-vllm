"""An eager torch / torch_npu composition of this operator, for use as a baseline.

There is no vendor kernel and no portable Triton implementation that builds on
this card, so the only honest comparison left is "what it costs to do the same
work with framework operators" -- which is what the fused kernel exists to
replace.

Two decisions keep the comparison fair rather than flattering:

  * where a vendor fused op computes the right thing, use it. `npu_rms_norm`
    with a unit gamma is this operator's weightless RMSNorm, so the Q
    normalisation is one op here rather than five; a baseline made deliberately
    naive would inflate the ratio.

    It is NOT bit-identical, which an earlier note here claimed on the strength
    of an 8-row check. Measured against an explicit `x * rsqrt(mean(x*x) + eps)`:
    8 rows agree exactly, 1024 rows differ in 8% of float32 words and 65536 rows
    in 7.5%, at a maximum relative difference of 3.5e-07 -- two to three float32
    ULP. The 8-row case returning exactly zero is not a small sample missing an
    8% effect; the implementation evidently takes a different path at size.
    After the bfloat16 rounding this operator applies, what survives is nothing
    at 1024 rows and 60 elements in 33.5M at 65536.

    This is a virtue rather than a flaw: the oracle uses the explicit form and
    the baseline the vendor op, so the RMSNorm is genuinely cross-checked
    between two implementations rather than one written twice.
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
