"""Standalone reproducer: `>>` on tl.uint16 lowers as an ARITHMETIC shift.

torch, torch_npu and triton only -- no FlagGems, no repo imports.

    python tools/repro_uint16_shift.py
"""

import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEV = "npu"
B = 128


# --------------------------------------------------------------- width sweep
@triton.jit
def k_shift_u8(x_ptr, out_ptr, SH: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    bits = tl.load(x_ptr + lane).to(tl.uint8, bitcast=True)
    tl.store(out_ptr + lane, (bits >> SH).to(tl.int32))


@triton.jit
def k_shift_u16(x_ptr, out_ptr, SH: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    bits = tl.load(x_ptr + lane).to(tl.uint16, bitcast=True)
    tl.store(out_ptr + lane, (bits >> SH).to(tl.int32))


@triton.jit
def k_shift_u32(x_ptr, out_ptr, SH: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    bits = tl.load(x_ptr + lane).to(tl.uint32, bitcast=True)
    tl.store(out_ptr + lane, (bits >> SH).to(tl.int32))


# ------------------------------------------- the workaround we ship today
@triton.jit
def k_shift_u16_widened(x_ptr, out_ptr, SH: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    bits = tl.load(x_ptr + lane).to(tl.uint16, bitcast=True)
    tl.store(out_ptr + lane, (bits.to(tl.int32) & 0xFFFF) >> SH)


# ----------------------------------- the real-world shape: fp16 radix bucket
@triton.jit
def k_fp16_bucket(x_ptr, out_ptr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    h = tl.load(x_ptr + lane).to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    sign_set = (bits & tl.full(bits.shape, 0x8000, tl.uint16)) != 0
    inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
    mapped = tl.where(sign_set, bits, inv)
    tl.store(out_ptr + lane, (mapped >> 5).to(tl.int32))


def run(kernel, x, sh=5):
    out = torch.zeros(B, dtype=torch.int32, device=DEV)
    kernel[(1,)](x, out, SH=sh, BLOCK=B)
    torch.npu.synchronize()
    return int(out[0])


def main():
    print(f"triton {triton.__version__} | torch {torch.__version__}")
    print()

    # Every input has its top bit set.  A logical >> 5 clears the top 5 bits;
    # an arithmetic >> 5 replicates the sign bit and yields all-ones.
    cases = [
        ("uint8 ", k_shift_u8, torch.full((B,), -2, dtype=torch.int8), 0xFE, 8),
        ("uint16", k_shift_u16, torch.full((B,), -2, dtype=torch.int16), 0xFFFE, 16),
        ("uint32", k_shift_u32, torch.full((B,), -2, dtype=torch.int32), 0xFFFFFFFE, 32),
    ]

    print(f"{'dtype':7} {'input':>12} {'got':>12} {'logical':>12} {'arith':>12}   verdict")
    print("-" * 78)
    for name, kernel, cpu_x, raw, width in cases:
        x = cpu_x.to(DEV)
        got = run(kernel, x)
        logical = raw >> 5
        arith = (raw - (1 << width)) >> 5  # sign-extended, then shifted
        verdict = ("LOGICAL (correct)" if got == logical else
                   "ARITHMETIC (defect)" if got == arith else "other")
        print(f"{name:7} {hex(raw):>12} {got:>12} {logical:>12} {arith:>12}   {verdict}")

    print()
    x16 = torch.full((B,), -2, dtype=torch.int16).to(DEV)
    got = run(k_shift_u16_widened, x16)
    print(f"workaround  (bits.to(tl.int32) & 0xFFFF) >> 5 : "
          f"got {got}, expected {0xFFFE >> 5} -> "
          f"{'OK' if got == 0xFFFE >> 5 else 'STILL WRONG'}")

    print()
    # Descending-order radix bucket of a NEGATIVE fp16: mapped keeps the raw
    # bits, whose top bit is set, so the bucket index is destroyed.
    xf = torch.full((B,), -1.5, dtype=torch.float32).to(DEV)
    got = run(k_fp16_bucket, xf, sh=5)
    raw = torch.tensor([-1.5], dtype=torch.float16).view(torch.int16).item() & 0xFFFF
    print(f"fp16 radix bucket of -1.5 (bits {hex(raw)}) : "
          f"got {got}, expected {raw >> 5} -> "
          f"{'OK' if got == raw >> 5 else 'WRONG -- every negative logit collapses'}")


if __name__ == "__main__":
    main()
