"""Standalone reproducer: `>>` on unsigned tl types lowers as an ARITHMETIC shift.

torch, torch_npu and triton only -- no FlagGems, no repo imports.

    python tools/repro_uint16_shift.py

Run it on every Ascend triton line you have; the verdict column is the answer.
"""

import os
import glob
import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

DEV = "npu"
B = 128


@triton.jit
def k_u8(x_ptr, out_ptr, SH: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    bits = tl.load(x_ptr + lane).to(tl.uint8, bitcast=True)
    tl.store(out_ptr + lane, (bits >> SH).to(tl.int32))


@triton.jit
def k_u16(x_ptr, out_ptr, SH: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    bits = tl.load(x_ptr + lane).to(tl.uint16, bitcast=True)
    tl.store(out_ptr + lane, (bits >> SH).to(tl.int32))


@triton.jit
def k_u32(x_ptr, out_ptr, SH: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    bits = tl.load(x_ptr + lane).to(tl.uint32, bitcast=True)
    tl.store(out_ptr + lane, (bits >> SH).to(tl.int32))


@triton.jit
def k_u16_widened(x_ptr, out_ptr, SH: tl.constexpr, BLOCK: tl.constexpr):
    """The workaround we ship: widen, mask off the sign extension, then shift."""
    lane = tl.arange(0, BLOCK)
    bits = tl.load(x_ptr + lane).to(tl.uint16, bitcast=True)
    tl.store(out_ptr + lane, (bits.to(tl.int32) & 0xFFFF) >> SH)


@triton.jit
def k_fp16_bucket(x_ptr, out_ptr, SH: tl.constexpr, BLOCK: tl.constexpr):
    """The real-world shape: a descending-order radix bucket of an fp16 value."""
    lane = tl.arange(0, BLOCK)
    h = tl.load(x_ptr + lane).to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    sign_set = (bits & tl.full(bits.shape, 0x8000, tl.uint16)) != 0
    inv = (~bits) & tl.full(bits.shape, 0x7FFF, tl.uint16)
    mapped = tl.where(sign_set, bits, inv)
    tl.store(out_ptr + lane, (mapped >> SH).to(tl.int32))


def run(kernel, x, sh):
    out = torch.zeros(B, dtype=torch.int32, device=DEV)
    kernel[(1,)](x, out, SH=sh, BLOCK=B)
    torch.npu.synchronize()
    return int(out[0])


def env():
    print(f"triton {triton.__version__}  at {os.path.dirname(triton.__file__)}")
    print(f"torch  {torch.__version__} | torch_npu {getattr(torch_npu, '__version__', '?')}")
    try:
        print(f"device {torch.npu.get_device_name(0)}")
    except Exception as exc:
        print(f"device <unavailable: {exc}>")
    for pat in ("/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
                "/usr/local/Ascend/*/version.cfg"):
        for path in sorted(glob.glob(pat))[:3]:
            try:
                body = open(path).read().strip().replace("\n", " ")
                print(f"CANN   {path}: {body[:120]}")
            except OSError:
                pass
    print()


def main():
    env()

    # Every input below has its top bit set.  A logical >> 5 clears the top
    # 5 bits; an arithmetic >> 5 replicates the sign bit and yields all-ones.
    SH = 5
    cases = [
        ("uint8", k_u8, torch.full((B,), -2, dtype=torch.int8), 0xFE, 8),
        ("uint16", k_u16, torch.full((B,), -2, dtype=torch.int16), 0xFFFE, 16),
        ("uint32", k_u32, torch.full((B,), -2, dtype=torch.int32), 0xFFFFFFFE, 32),
    ]

    print("top bit SET -- this is where the two shifts differ")
    print(f"{'dtype':7} {'input':>12} {'got':>12} {'logical':>12} {'arith':>12}   verdict")
    print("-" * 78)
    for name, kernel, cpu_x, raw, width in cases:
        got = run(kernel, cpu_x.to(DEV), SH)
        logical = raw >> SH
        arith = (raw - (1 << width)) >> SH
        verdict = ("LOGICAL (correct)" if got == logical else
                   "ARITHMETIC (defect)" if got == arith else "other")
        print(f"{name:7} {hex(raw):>12} {got:>12} {logical:>12} {arith:>12}   {verdict}")

    # Control: the same shifts on inputs whose top bit is CLEAR must all pass.
    # If one of these fails the defect is not about sign extension at all.
    print()
    print("top bit CLEAR -- control, every line must read correct")
    print(f"{'dtype':7} {'input':>12} {'got':>12} {'expected':>12}   verdict")
    print("-" * 62)
    controls = [
        ("uint8", k_u8, torch.tensor([0x7E] * B, dtype=torch.int8), 0x7E),
        ("uint16", k_u16, torch.tensor([0x7FFE] * B, dtype=torch.int16), 0x7FFE),
        ("uint32", k_u32, torch.tensor([0x7FFFFFFE] * B, dtype=torch.int32), 0x7FFFFFFE),
    ]
    for name, kernel, cpu_x, raw in controls:
        got = run(kernel, cpu_x.to(DEV), SH)
        exp = raw >> SH
        print(f"{name:7} {hex(raw):>12} {got:>12} {exp:>12}   "
              f"{'correct' if got == exp else 'WRONG'}")

    print()
    got = run(k_u16_widened, torch.full((B,), -2, dtype=torch.int16).to(DEV), SH)
    exp = 0xFFFE >> SH
    print(f"workaround  (bits.to(tl.int32) & 0xFFFF) >> 5 : got {got}, "
          f"expected {exp} -> {'OK' if got == exp else 'STILL WRONG'}")

    # Descending-order radix bucket of a NEGATIVE fp16: mapped keeps the raw
    # bits, whose top bit is set, so the bucket index is destroyed.
    got = run(k_fp16_bucket, torch.full((B,), -1.5, dtype=torch.float32).to(DEV), SH)
    raw = torch.tensor([-1.5], dtype=torch.float16).view(torch.int16).item() & 0xFFFF
    print(f"fp16 radix bucket of -1.5 (bits {hex(raw)})  : got {got}, "
          f"expected {raw >> SH} -> "
          f"{'OK' if got == raw >> SH else 'WRONG -- every negative logit collapses'}")


if __name__ == "__main__":
    main()
