"""A block-private coarse histogram stage for the BW1000 feasibility probe."""

import triton
import triton.language as tl


@triton.jit
def private_hist256(
    logits,
    starts,
    ends,
    scratch,
    stride0,
    CHUNKS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    row = tile // CHUNKS
    chunk = tile % CHUNKS
    start = tl.load(starts + row)
    end = tl.load(ends + row)
    pos = chunk * BLOCK + tl.arange(0, BLOCK)
    valid = pos < end - start
    x = tl.load(
        logits + row * stride0 + start + pos,
        mask=valid,
        other=0.0,
    )
    # Exactly the shipped STEP-0 float16 key, coarsened from 11 to 8 bits.
    half = x.to(tl.float16)
    bits = half.to(tl.uint16, bitcast=True)
    mapped = tl.where((bits & 0x8000) != 0, bits, (~bits) & 0x7FFF)
    key = (mapped >> 8).to(tl.int32)
    bins = tl.histogram(key, 256, mask=valid)
    tl.store(scratch + tile * 256 + tl.arange(0, 256), bins)
