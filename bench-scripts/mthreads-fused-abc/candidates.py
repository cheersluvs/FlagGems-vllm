# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Experimental B/B1/C kernels; not imported by the production operator."""

import triton
import triton.language as tl


@triton.jit
def _rotate(rope, cos, sin, ROWS: tl.constexpr):
    pairs = tl.reshape(rope, ROWS, 32, 2)
    even, odd = tl.split(pairs)
    return tl.reshape(
        tl.join(even * cos - odd * sin, even * sin + odd * cos), ROWS, 64
    ).to(tl.bfloat16)


@triton.jit
def _q_heads(q, positions, cs, eps, tok, group, H: tl.constexpr, HPP: tl.constexpr):
    heads = group * HPP + tl.arange(0, HPP).to(tl.int64)
    good = heads < H
    off = tl.arange(0, 512)
    base = q + (tok * H + heads) * 512
    value = tl.load(base[:, None] + off[None, :], good[:, None], other=0).to(tl.float32)
    inv = tl.rsqrt(tl.sum(value * value, 1) / 512 + eps)
    tl.store(
        base[:, None] + off[None, :],
        (value * inv[:, None]).to(tl.bfloat16),
        good[:, None] & (off[None, :] < 448),
    )
    r = tl.arange(0, 64)
    rope = (
        tl.load(base[:, None] + 448 + r[None, :], good[:, None], other=0).to(tl.float32)
        * inv[:, None]
    )
    pos = tl.load(positions + tok)
    half = tl.arange(0, 32)
    # One token's cos/sin broadcast over adjacent heads, not separate tokens.
    cos = tl.load(cs + pos * 64 + half)
    sin = tl.load(cs + pos * 64 + 32 + half)
    rotated = _rotate(rope, cos[None, :], sin[None, :], HPP)
    tl.store(base[:, None] + 448 + r[None, :], rotated, good[:, None])
    return cos, sin


@triton.jit
def _kv_insert(
    kv,
    cache,
    slots,
    tok,
    good,
    cos,
    sin,
    stride,
    I,
    BS: tl.constexpr,
    ROWS: tl.constexpr,
):
    r = tl.arange(0, 64)
    base = kv + tok * 512
    rope = tl.load(base[:, None] + 448 + r[None, :], good[:, None], other=0).to(
        tl.float32
    )
    rope = _rotate(rope, cos, sin, ROWS)
    ins = good & (tok < I)
    slot = tl.load(slots + tok, ins, other=-1)
    ins = ins & (slot >= 0)
    block = slot // BS
    at = slot % BS
    data = block * stride + at * 576
    scales = block * stride + BS * 576 + at * 8
    ptr = (cache + data + 448).to(tl.pointer_type(tl.bfloat16))
    tl.store(ptr[:, None] + r[None, :], rope, ins[:, None])
    for b in tl.static_range(7):
        value = tl.load(base[:, None] + b * 64 + r[None, :], ins[:, None], other=0).to(
            tl.float32
        )
        maximum = tl.maximum(tl.max(tl.abs(value), 1), 1e-4)
        exponent = tl.ceil(tl.log2(maximum / 448.0))
        scale = tl.exp2(exponent)
        quant = tl.clamp(value / scale[:, None], -448.0, 448.0)
        tl.store(
            cache + data[:, None] + b * 64 + r[None, :],
            quant.to(tl.float8e4nv).to(tl.uint8, bitcast=True),
            ins[:, None],
        )
        encoded = tl.maximum(tl.minimum(exponent + 127.0, 255.0), 0.0)
        tl.store(cache + scales + b, encoded.to(tl.uint8), ins)
    tl.store(cache + scales + 7, tl.full((ROWS,), 0, tl.uint8), ins)


@triton.jit
def head_kernel(
    q,
    kv,
    cache,
    slots,
    positions,
    cs,
    eps,
    N,
    stride,
    I,
    H: tl.constexpr,
    BS: tl.constexpr,
    HPP: tl.constexpr,
    KV_TPP: tl.constexpr,
    MERGE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    GROUPS: tl.constexpr = triton.cdiv(H, HPP)
    q_programs = N * GROUPS
    if MERGE:
        merged_tok = pid // GROUPS
        merged_group = pid % GROUPS
        merged_cos, merged_sin = _q_heads(
            q, positions, cs, eps, merged_tok, merged_group, H, HPP
        )
        if merged_group == GROUPS - 1:
            # Includes the last Q head(s), THEN this token's one KV; no Q replaced.
            merged_tok_vector = tl.full((1,), merged_tok, tl.int64)
            merged_good = tl.full((1,), True, tl.int1)
            _kv_insert(
                kv,
                cache,
                slots,
                merged_tok_vector,
                merged_good,
                merged_cos[None, :],
                merged_sin[None, :],
                stride,
                I,
                BS,
                1,
            )
    else:
        if pid < q_programs:
            # Separate names prevent a scalar/vector SSA merge at the branch join.
            q_tok = pid // GROUPS
            q_group = pid % GROUPS
            _q_heads(q, positions, cs, eps, q_tok, q_group, H, HPP)
        else:
            kv_tok = (pid - q_programs) * KV_TPP + tl.arange(0, KV_TPP).to(tl.int64)
            kv_good = kv_tok < N
            kv_pos = tl.load(positions + kv_tok, kv_good, other=0)
            kv_half = tl.arange(0, 32)
            kv_cos = tl.load(
                cs + kv_pos[:, None] * 64 + kv_half[None, :],
                kv_good[:, None],
                other=0,
            )
            kv_sin = tl.load(
                cs + kv_pos[:, None] * 64 + 32 + kv_half[None, :],
                kv_good[:, None],
                other=0,
            )
            _kv_insert(
                kv,
                cache,
                slots,
                kv_tok,
                kv_good,
                kv_cos,
                kv_sin,
                stride,
                I,
                BS,
                KV_TPP,
            )
