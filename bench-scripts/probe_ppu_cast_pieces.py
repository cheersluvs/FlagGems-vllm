"""Which instruction of #1116's software fp8 cast does ppu-llc reject?

The shim attaches and the compile reaches the vendor's binary stage, where
ppu-llc fails to parse its own input at line 1 with `mismatched input 'ppu'`:
something in the generated assembly kept the `ppu.` mnemonic prefix that the
.tix -> .tix.trans step is supposed to translate. The message names no
instruction, so this probe compiles the cast's ingredients one at a time and
reports which ones the toolchain accepts. Each piece is one micro-kernel; the
first failure names the construct to avoid (or to report).

    REPO=/path/to/worktree PYTHONPATH=$REPO/src:$PYTHONPATH \
        python3 bench-scripts/probe_ppu_cast_pieces.py
"""

import os
import sys
import traceback

REPO = os.environ.get("REPO", os.getcwd())
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

N = 256


@triton.jit
def k_control(src, dst, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    x = tl.load(src + o, mask=m, other=0.0)
    tl.store(dst + o, x.to(tl.int32, bitcast=True), mask=m)


@triton.jit
def k_var_shift(src, dst, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    b = tl.load(src + o, mask=m, other=0.0).to(tl.int32, bitcast=True)
    k = (b & 7) + 20                      # per-lane variable shift amount
    tl.store(dst + o, (b >> k) + (b << (k - 19)), mask=m)


@triton.jit
def k_minmax(src, dst, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    b = tl.load(src + o, mask=m, other=0.0).to(tl.int32, bitcast=True)
    tl.store(dst + o, tl.minimum(tl.maximum(b, 0), 6), mask=m)


@triton.jit
def k_where_chain(src, dst, n, BLOCK: tl.constexpr):
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    b = tl.load(src + o, mask=m, other=0.0).to(tl.int32, bitcast=True)
    r = tl.where(b > 0, b, -b)
    r = tl.where(b == 0, 0x7E, r)
    r = tl.where((b & 0x7FFFFFFF) > 0x7F800000, 0x7F, r)
    tl.store(dst + o, r, mask=m)


@triton.jit
def k_magic_rne(src, dst, n, BLOCK: tl.constexpr):
    """The override's approach: rounding by a float add, no per-lane shift."""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    x = tl.load(src + o, mask=m, other=0.0)
    y = x * 512.0 + 8388608.0 - 8388608.0
    tl.store(dst + o, y.to(tl.int32), mask=m)


@triton.jit
def k_fp8_bitcast(src, dst, n, BLOCK: tl.constexpr):
    """fp8 exists only as a bitcast in the IR -- no fp8 arithmetic, no fp8 memory."""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    u = tl.load(src + o, mask=m, other=0.0).to(tl.int32, bitcast=True).to(tl.uint8)
    f8 = u.to(tl.float8e4nv, bitcast=True)
    tl.store(dst + o, f8.to(tl.uint8, bitcast=True), mask=m)


@triton.jit
def k_fp8_store(src, dst8, n, BLOCK: tl.constexpr):
    """A real fp8 pointer: the store the operator actually performs."""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    u = tl.load(src + o, mask=m, other=0.0).to(tl.int32, bitcast=True).to(tl.uint8)
    tl.store(dst8 + o, u.to(tl.float8e4nv, bitcast=True), mask=m)


@triton.jit
def k_full_downcast(src, dst, n, BLOCK: tl.constexpr):
    """What failed: the whole #1116 software cast behind `.to(tl.float8e4nv)`."""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    x = tl.load(src + o, mask=m, other=0.0)
    tl.store(dst + o, x.to(tl.float8e4nv).to(tl.uint8, bitcast=True), mask=m)


@triton.jit
def k_upcast(src8, dst, n, BLOCK: tl.constexpr):
    """Informational: the direction #1116 implements in C++, not in the frontend."""
    o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = o < n
    f8 = tl.load(src8 + o, mask=m)
    tl.store(dst + o, f8.to(tl.float32), mask=m)


def main():
    import flaggems_vllm
    import ppu_fp8_shim

    dev = flaggems_vllm.device
    fn = flaggems_vllm.runtime.torch_device_fn
    tdev = dev if isinstance(dev, torch.device) else torch.device(dev)

    log = []
    keep = None
    try:
        keep = ppu_fp8_shim.preserve_vendor_asm(log)
        ppu_fp8_shim._patch_backend_options(log)
        ppu_fp8_shim._patch_semantic_cast(log)
    except Exception as e:
        print("  shim could not attach: {}: {}".format(type(e).__name__, e))
    for line in log:
        print("  " + line)

    x = torch.randn(N, device=tdev)
    i32 = torch.zeros(N, dtype=torch.int32, device=tdev)
    u8 = torch.zeros(N, dtype=torch.uint8, device=tdev)
    f8 = torch.zeros(N, dtype=torch.float8_e4m3fn, device=tdev)
    grid = (1, )

    pieces = [
        ("control  f32 -> i32 bitcast", k_control, (x, i32, N)),
        ("per-lane variable shift", k_var_shift, (x, i32, N)),
        ("int minimum / maximum", k_minmax, (x, i32, N)),
        ("three chained tl.where", k_where_chain, (x, i32, N)),
        ("magic-number RNE (override style)", k_magic_rne, (x, i32, N)),
        ("u8 -> fp8e4nv bitcast, no fp8 memory", k_fp8_bitcast, (x, u8, N)),
        ("store to a real fp8 pointer", k_fp8_store, (x, f8, N)),
        ("FULL software downcast", k_full_downcast, (x, u8, N)),
        ("fp8 -> f32 upcast (C++ in #1116)", k_upcast, (f8, x.clone(), N)),
    ]
    print("\n  {:<40} {}".format("piece", "result"))
    print("  " + "-" * 74)
    failed_first = None
    for name, kern, args in pieces:
        try:
            kern[grid](*args, BLOCK=N, num_warps=1)
            fn.synchronize()
            print("  {:<40} ok".format(name))
        except Exception as e:
            first = [ln for ln in str(e).splitlines() if ln.strip()]
            msg = first[0][:70] if first else type(e).__name__
            print("  {:<40} FAIL  {}".format(name, msg))
            if failed_first is None:
                failed_first = (name, traceback.format_exc())
        fn.empty_cache()

    if keep and os.path.isdir(keep):
        tix = sorted(f for f in os.listdir(keep) if f.endswith(".trans"))
        if tix:
            path = os.path.join(keep, tix[0])
            print("\n  rejected assembly {} -- first 12 lines:".format(path))
            with open(path, errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= 12:
                        break
                    print("    {:>3}| {}".format(i + 1, line.rstrip()[:150]))
            body = open(path, errors="replace").read()
            hits = [ln.strip() for ln in body.splitlines() if "ppu." in ln]
            print("  lines still carrying the untranslated `ppu.` prefix: {}".format(len(hits)))
            for ln in hits[:8]:
                print("    {}".format(ln[:150]))
    print("\n[RESULT] PIECES_DONE")


try:
    main()
except Exception:
    traceback.print_exc()
    print("\n[RESULT] FAILED")
sys.stdout.flush()
