# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ascend override for the fused DeepSeek-V4 qnorm/RoPE/quant/insert kernel.

The arithmetic is the generic implementation's throughout. Four things are
expressed differently, each because the toolchain rejects the generic form:

1. FP8 conversion goes through `_f32_to_e4m3_bits`, not `.to(tl.float8e4nv)`.
2. The UE8M0 scale is read off the exponent bits by `_ue8m0_scale`, not built
   with log2/ceil/exp2 and a float-to-integer conversion.
3. The bf16 half of each cache token is reached through a bfloat16 view passed
   in by the wrapper, not by casting a pointer's element type.
4. RoPE pairs are addressed with a 2-D offset, so they need no reshape.

Each is explained at its definition or use site. Points 2, 3 and 4 also each
surfaced as a compiler failure that reports the wrong thing -- a missing output
file, or nothing at all -- so read those notes before reformulating any of them.

`tl.float8e4nv` cannot be compiled for this card, but for a different reason than
on T-Head. There is no capability gate to argue with — BiShengIR does not know the
type at all:

    error: 'hivm.hir.vcast' op currently don't support cast float_to_UNKNOWN_rintmode
    error: unrecognized float type: 'f8E4M3FN'

So the conversion is done with integer operations, which the backend compiles
without complaint. Verified bit-identical to `torch.float8_e4m3fn` on the card at
block widths 256 through 2048; the operator uses 512.

Ascend's Unified Buffer is the constraint to watch when changing this: the encoder
keeps roughly fifteen live intermediates, so a 4096-wide block asks for 240 KB
against the 192 KB available and fails to compile with `ub overflow, requires
1966080 bits while 1572864 bits available`. That is a loud compile-time failure
rather than a silent one, but it caps how much this kernel can be widened.

This file can go away once BiShengIR lowers `f8E4M3FN`, AICore can select a
scalar float-to-integer conversion, Triton can bitcast a pointer's element type
here, and `[ROPE_DIM] -> [HALF_ROPE_DIM, 2]` survives the shape pipeline.
"""

import torch
import triton
import triton.language as tl

# The runtime rejects a launch whose program count exceeds this, reporting it as
# an invalid `coreDim`. It is a launch-API limit, not a hardware occupancy one.
MAX_PROGRAMS_PER_LAUNCH = 65535

# (token, head) units handled by one program. A unit moves about a kilobyte,
# far too little to cover this backend's per-program cost, so the units are
# batched. Chosen from a measured copy sweep on the card: 1 unit per program
# reaches 51 GB/s, 2 -> 104, 4 -> 203, 8 -> 288, and it is flat from there to
# 64. Overridable for re-tuning, but 8 is where the curve turns over.
UNITS_PER_PROGRAM = 8


@triton.jit
def _f32_to_e4m3_bits(x):
    """float32 -> OCP E4M3 (float8_e4m3fn) bit pattern, round-to-nearest-even.

    Handles subnormals (m * 2^-9 for m in 1..7) and saturates to 448. Note that
    exponent field 15 with mantissa 0 is the legal value 256, not an overflow --
    only e > 15, or e == 15 with mantissa 7 (the NaN encoding), may saturate.
    """
    b = x.to(tl.int32, bitcast=True)
    sign = (b >> 24) & 0x80
    mag = b & 0x7FFFFFFF
    sig = (mag & 0x7FFFFF) | 0x800000
    e = (mag >> 23) - 120

    m_n = (mag >> 20) & 0x7
    round_n = (mag >> 19) & 1
    sticky_n = (mag & 0x7FFFF) != 0
    m_n = m_n + tl.where((round_n == 1) & (sticky_n | ((m_n & 1) == 1)), 1, 0)
    e_n = e + tl.where(m_n > 7, 1, 0)
    m_n = tl.where(m_n > 7, 0, m_n)

    sh = tl.minimum(tl.maximum(21 - e, 0), 31)
    m_s = sig >> sh
    round_s = (sig >> tl.maximum(sh - 1, 0)) & 1
    sticky_s = (sig & ((1 << tl.maximum(sh - 1, 0)) - 1)) != 0
    m_s = m_s + tl.where((round_s == 1) & (sticky_s | ((m_s & 1) == 1)), 1, 0)

    v = tl.where(e >= 1, (e_n << 3) | m_n, m_s)
    v = tl.where((e_n > 15) | ((e_n == 15) & (m_n == 7)), 0x7E, v)
    v = tl.where(mag == 0, 0, v)
    return (sign | v).to(tl.uint8)


@triton.jit
def _ue8m0_scale(raw_scale):
    """UE8M0 scale for a positive float: 2^ceil(log2(raw_scale)), plus the byte
    that encodes it -- which is exactly that power of two's biased exponent.

    Read straight off the exponent bits rather than going through
    log2 / ceil / exp2 and a float-to-integer conversion. The conversion is the
    reason this exists: the value is a per-block scalar, and AICore's scalar
    unit has no float-to-integer instruction, so `encoded_scale.to(tl.uint8)`
    fails instruction selection in bisheng with

        fatal error: error in backend: Cannot select: i64 = fp_to_uint

    (wrapped in the NaN and lower-bound selects Triton emits for a saturating
    conversion). The failure is reported as a missing kernel.o, because hivmc
    exits 0 after bisheng fails.

    raw_scale is always positive here -- block_max is floored at 1e-4 -- so the
    sign bit needs no handling.
    """
    bits = raw_scale.to(tl.int32, bitcast=True)
    biased_exp = (bits >> 23) & 0xFF
    # Ceil rather than floor: any set mantissa bit means the next power of two.
    code = biased_exp + tl.where((bits & 0x7FFFFF) != 0, 1, 0)
    scale = (code << 23).to(tl.float32, bitcast=True)
    stored = tl.minimum(tl.maximum(code, 0), 255)
    return scale, stored


@triton.jit
def _process_one(
    pid,
    q,
    kv,
    k_cache,
    k_cache_bf16,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps,
    cache_block_size: tl.constexpr,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    kv_block_stride,
    num_tokens_insert: tl.constexpr,
):
    """One (token, head) unit of work -- the generic kernel body, unchanged.

    Split out as a device function so the launcher can run several of these per
    program. Its early returns are preserved: a `return` here leaves this unit,
    not the program, which is what lets the body stay identical to the generic
    implementation instead of being restructured into nested conditionals. That
    matters on this backend, where in-loop control flow is what breaks the UB
    allocator.
    """
    HEAD_DIM: tl.constexpr = 512
    NOPE_DIM: tl.constexpr = 448
    ROPE_DIM: tl.constexpr = 64
    HALF_ROPE_DIM: tl.constexpr = 32
    QUANT_BLOCK: tl.constexpr = 64
    NUM_QUANT_BLOCKS: tl.constexpr = NOPE_DIM // QUANT_BLOCK  # 7
    SCALE_BYTES_PER_TOKEN: tl.constexpr = NUM_QUANT_BLOCKS + 1  # 8 (7 real + 1 pad)
    TOKEN_DATA_BYTES: tl.constexpr = NOPE_DIM + 2 * ROPE_DIM  # 576
    FP8_MAX: tl.constexpr = 448.0

    blocks_per_token = num_heads + 1
    token_idx = pid // blocks_per_token
    if token_idx >= num_tokens:
        return
    slot_idx = pid % blocks_per_token
    is_kv = slot_idx == num_heads
    if is_kv and token_idx >= num_tokens_insert:  # no need to insert
        return
    q_base = q + (token_idx * num_heads + slot_idx) * HEAD_DIM
    kv_base = kv + token_idx * HEAD_DIM
    offset = tl.arange(0, HEAD_DIM)
    mask_nope = offset < NOPE_DIM
    offset_half_rope = tl.arange(0, HALF_ROPE_DIM)
    offset_quant = tl.arange(0, QUANT_BLOCK)
    # The RoPE pairs are addressed with a 2-D offset, so they arrive already
    # shaped [HALF_ROPE_DIM, 2] and need no reshape. BiShengIR rejects turning
    # [ROPE_DIM] into [HALF_ROPE_DIM, 2] here (`cannot align 0 axis` on the
    # expand_shape, `collapsing non-contiguous dims` on the way back), and
    # taking the pairs with 1-D stride-2 offsets instead aborts the compiler in
    # InterleaveOptimization.cpp. This form needs neither.
    offset_pair = offset_half_rope[:, None] * 2 + tl.arange(0, 2)[None, :]
    if not is_kv:
        # load q
        q_blk = tl.load(q_base + offset).to(tl.float32)  # [NOPE_DIM]
        q_blk_rope = tl.load(q_base + NOPE_DIM + offset_pair).to(
            tl.float32
        )  # [HALF_ROPE_DIM, 2]
        # RMSNorm with no weight
        variance = tl.sum(q_blk * q_blk) / HEAD_DIM
        rsqrt = tl.rsqrt(variance + eps)
        q_blk = q_blk * rsqrt
        # store q nope
        tl.store(q_base + offset, q_blk.to(tl.bfloat16), mask=mask_nope)  # [NOPE_DIM]
        qkv_blk_rope = q_blk_rope * rsqrt
    else:
        # load kv rope
        qkv_blk_rope = tl.load(kv_base + NOPE_DIM + offset_pair).to(
            tl.float32
        )  # [HALF_ROPE_DIM, 2]
    # load cos/sin
    position_id = tl.load(position_ids + token_idx)  # i64
    cs_base = cos_sin_cache + position_id * ROPE_DIM
    cos_blk = tl.load(cs_base + offset_half_rope)  # [HALF_ROPE_DIM], f32
    sin_blk = tl.load(
        cs_base + offset_half_rope + HALF_ROPE_DIM
    )  # [HALF_ROPE_DIM], f32
    # ROPE
    even_blk, odd_blk = tl.split(qkv_blk_rope)  # [HALF_ROPE_DIM], f32
    new_even_blk = even_blk * cos_blk - odd_blk * sin_blk
    new_odd_blk = even_blk * sin_blk + odd_blk * cos_blk
    qkv_blk_rope = tl.join(new_even_blk, new_odd_blk).to(
        tl.bfloat16
    )  # [HALF_ROPE_DIM, 2]
    if not is_kv:
        # store q rope
        tl.store(q_base + NOPE_DIM + offset_pair, qkv_blk_rope)  # [HALF_ROPE_DIM, 2]
        return
    # load slot
    slot_id = tl.load(slot_mapping + token_idx)  # i64
    if slot_id < 0:
        return
    block_idx = slot_id // cache_block_size
    pos_in_block = slot_id % cache_block_size
    block_base = k_cache + block_idx * kv_block_stride
    token_fp8_ptr = block_base + pos_in_block * TOKEN_DATA_BYTES
    # Ascend's Triton rejects casting a pointer between element widths --
    # `Casting pointers with unmatched bitwidth!` -- so the RoPE region is
    # reached through a bf16 view passed in from the host. Every byte offset
    # here is even (block stride 37376, 576 per token, 448 NoPE), so halving
    # them is exact.
    token_bf16_idx = (
        block_idx * kv_block_stride + pos_in_block * TOKEN_DATA_BYTES + NOPE_DIM
    ) // 2
    token_scale_ptr = (
        block_base
        + cache_block_size * TOKEN_DATA_BYTES
        + pos_in_block * SCALE_BYTES_PER_TOKEN
    )
    # store kv rope
    tl.store(
        k_cache_bf16 + token_bf16_idx + offset_pair, qkv_blk_rope
    )  # [HALF_ROPE_DIM, 2]
    # quantization of kv nope
    # unroll the quantization loop and co-issue loads for better performance
    kv_quant_blk0 = tl.load(kv_base + offset_quant)
    kv_quant_blk1 = tl.load(kv_base + QUANT_BLOCK + offset_quant)
    kv_quant_blk2 = tl.load(kv_base + 2 * QUANT_BLOCK + offset_quant)
    kv_quant_blk3 = tl.load(kv_base + 3 * QUANT_BLOCK + offset_quant)
    kv_quant_blk4 = tl.load(kv_base + 4 * QUANT_BLOCK + offset_quant)
    kv_quant_blk5 = tl.load(kv_base + 5 * QUANT_BLOCK + offset_quant)
    kv_quant_blk6 = tl.load(kv_base + 6 * QUANT_BLOCK + offset_quant)
    # quant group 0
    qblock_idx = 0
    kv_quant_blk = kv_quant_blk0.to(tl.float32)
    block_max = tl.max(tl.abs(kv_quant_blk), axis=0)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX)), taken off the exponent bits
    raw_scale = block_max / FP8_MAX
    scale, scale_code = _ue8m0_scale(raw_scale)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_uint8 = _f32_to_e4m3_bits(x_clamped)
    # store quantized data
    tl.store(token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant, x_uint8)
    # store scale: stored_value = exponent + 127 (bias)
    tl.store(token_scale_ptr + qblock_idx, scale_code.to(tl.uint8))

    # quant group 1
    qblock_idx = 1
    kv_quant_blk = kv_quant_blk1.to(tl.float32)
    block_max = tl.max(tl.abs(kv_quant_blk), axis=0)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX)), taken off the exponent bits
    raw_scale = block_max / FP8_MAX
    scale, scale_code = _ue8m0_scale(raw_scale)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_uint8 = _f32_to_e4m3_bits(x_clamped)
    # store quantized data
    tl.store(token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant, x_uint8)
    # store scale: stored_value = exponent + 127 (bias)
    tl.store(token_scale_ptr + qblock_idx, scale_code.to(tl.uint8))

    # quant group 2
    qblock_idx = 2
    kv_quant_blk = kv_quant_blk2.to(tl.float32)
    block_max = tl.max(tl.abs(kv_quant_blk), axis=0)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX)), taken off the exponent bits
    raw_scale = block_max / FP8_MAX
    scale, scale_code = _ue8m0_scale(raw_scale)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_uint8 = _f32_to_e4m3_bits(x_clamped)
    # store quantized data
    tl.store(token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant, x_uint8)
    # store scale: stored_value = exponent + 127 (bias)
    tl.store(token_scale_ptr + qblock_idx, scale_code.to(tl.uint8))

    # quant group 3
    qblock_idx = 3
    kv_quant_blk = kv_quant_blk3.to(tl.float32)
    block_max = tl.max(tl.abs(kv_quant_blk), axis=0)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX)), taken off the exponent bits
    raw_scale = block_max / FP8_MAX
    scale, scale_code = _ue8m0_scale(raw_scale)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_uint8 = _f32_to_e4m3_bits(x_clamped)
    # store quantized data
    tl.store(token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant, x_uint8)
    # store scale: stored_value = exponent + 127 (bias)
    tl.store(token_scale_ptr + qblock_idx, scale_code.to(tl.uint8))

    # quant group 4
    qblock_idx = 4
    kv_quant_blk = kv_quant_blk4.to(tl.float32)
    block_max = tl.max(tl.abs(kv_quant_blk), axis=0)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX)), taken off the exponent bits
    raw_scale = block_max / FP8_MAX
    scale, scale_code = _ue8m0_scale(raw_scale)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_uint8 = _f32_to_e4m3_bits(x_clamped)
    # store quantized data
    tl.store(token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant, x_uint8)
    # store scale: stored_value = exponent + 127 (bias)
    tl.store(token_scale_ptr + qblock_idx, scale_code.to(tl.uint8))

    # quant group 5
    qblock_idx = 5
    kv_quant_blk = kv_quant_blk5.to(tl.float32)
    block_max = tl.max(tl.abs(kv_quant_blk), axis=0)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX)), taken off the exponent bits
    raw_scale = block_max / FP8_MAX
    scale, scale_code = _ue8m0_scale(raw_scale)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_uint8 = _f32_to_e4m3_bits(x_clamped)
    # store quantized data
    tl.store(token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant, x_uint8)
    # store scale: stored_value = exponent + 127 (bias)
    tl.store(token_scale_ptr + qblock_idx, scale_code.to(tl.uint8))

    # quant group 6
    qblock_idx = 6
    kv_quant_blk = kv_quant_blk6.to(tl.float32)
    block_max = tl.max(tl.abs(kv_quant_blk), axis=0)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX)), taken off the exponent bits
    raw_scale = block_max / FP8_MAX
    scale, scale_code = _ue8m0_scale(raw_scale)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_uint8 = _f32_to_e4m3_bits(x_clamped)
    # store quantized data
    tl.store(token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant, x_uint8)
    # store scale: stored_value = exponent + 127 (bias)
    tl.store(token_scale_ptr + qblock_idx, scale_code.to(tl.uint8))

    # padding of scale
    tl.store(token_scale_ptr + NUM_QUANT_BLOCKS, tl.zeros((), dtype=tl.uint8))


@triton.jit
def fused_qnorm_rope_kv_insert_kernel(
    q,
    kv,
    k_cache,
    k_cache_bf16,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps,
    cache_block_size: tl.constexpr,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    kv_block_stride,
    pid_offset,
    total_units,
    num_tokens_insert: tl.constexpr,
    UNITS_PER_PROGRAM: tl.constexpr,
):
    """Run UNITS_PER_PROGRAM (token, head) units per program.

    One unit per program -- the generic shape -- costs about 76 ns each here and
    is the operator's entire runtime; the work per unit is only a kilobyte, so
    the fixed part dominates. Measured on this card with an isolated copy: 1 row
    per program reaches 51 GB/s, 2 rows 104, 4 rows 203, 8 rows 288, after which
    it is flat to 64 rows. num_warps is not a lever -- 1, 2 and 4 are identical.
    """
    base = (tl.program_id(0).to(tl.int64) + pid_offset) * UNITS_PER_PROGRAM
    for k in tl.static_range(UNITS_PER_PROGRAM):
        pid = base + k
        if pid < total_units:
            _process_one(
                pid,
                q,
                kv,
                k_cache,
                k_cache_bf16,
                slot_mapping,
                position_ids,
                cos_sin_cache,
                eps,
                cache_block_size,
                num_tokens,
                num_heads,
                kv_block_stride,
                num_tokens_insert,
            )


def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    cache_block_size: int,
):
    """
    Horizontally-fused DeepseekV4-MLA: per-head RMSNorm + GPT-J RoPE for Q, and
    GPT-J RoPE + UE8M0 FP8 quant + paged cache insert for KV, all in one kernel
    launch.
    K Cache block layout (block_size=64 tokens):
    - First 64 * 576 = 36864 bytes: Token data
      - Each token: 448 bytes (fp8) + 128 bytes (bf16)
    - Next 64 * 8 = 512 bytes: Scales
      - Each token: 8 bytes (uint8 scales, 7 real + 1 padding)
    - Padded to multiple of 576

    Args:
        q: [num_tokens, num_heads, 512], bfloat16, in place
        kv: [num_tokens, 512], bfloat16, read-only
        k_cache: [num_blocks, block_bytes], uint8
        slot_mapping: [num_tokens_insert], i64
        position_ids: [num_tokens], i64
        cos_sin_cache: [max_pos, 64], fp32
        eps: used in RMSNorm
        cache_block_size: tokens per paged-cache block
    """
    assert q.is_contiguous() and kv.is_contiguous()
    num_tokens, num_heads, head_dims = q.shape
    assert head_dims == 512
    assert kv.shape == (num_tokens, 512)
    assert q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16
    assert k_cache.dtype == torch.uint8
    assert slot_mapping.dim() == 1
    num_tokens_insert = slot_mapping.shape[0]
    assert num_tokens_insert <= num_tokens
    assert slot_mapping.dtype == torch.int64
    assert position_ids.shape == (num_tokens,)
    assert position_ids.dtype == torch.int64
    assert cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == 64
    assert cos_sin_cache.dtype == torch.float32

    assert k_cache.is_contiguous()
    k_cache_bf16 = k_cache.view(torch.bfloat16)

    # One program per (token, head), as in the generic implementation -- but the
    # runtime refuses a launch wider than MAX_PROGRAMS_PER_LAUNCH:
    #
    #   KernelLaunch failed because value 532480 for parameter coreDim is
    #   invalid. Expected value: less than or equal to 65535.
    #
    # which 8192 tokens at 64 heads already exceeds eightfold. The work is
    # issued in chunks and each program adds its chunk's offset to its id.
    total_units = num_tokens * (num_heads + 1)
    total_programs = triton.cdiv(total_units, UNITS_PER_PROGRAM)
    for pid_offset in range(0, total_programs, MAX_PROGRAMS_PER_LAUNCH):
        grid = min(MAX_PROGRAMS_PER_LAUNCH, total_programs - pid_offset)
        fused_qnorm_rope_kv_insert_kernel[(grid,)](
            q,
            kv,
            k_cache,
            k_cache_bf16,
            slot_mapping,
            position_ids,
            cos_sin_cache,
            eps,
            cache_block_size,
            num_tokens,
            num_heads,
            k_cache.stride(0),
            pid_offset,
            total_units,
            num_tokens_insert,
            UNITS_PER_PROGRAM,
            num_warps=1,
            num_stages=2,
        )
