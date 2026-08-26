#!/usr/bin/env python3
"""MTT S5000 preflight for top_k_per_row_{prefill,decode}.

Answers, in ONE round-trip, every environment question that decides whether a
run on this card means anything. Written for the no-SSH workflow: each check is
isolated so a hard failure still leaves a useful log above it.

Stages (MUSA errors are sticky -- one bad kernel poisons the context, so kernel
launches are opt-in and one-at-a-time):

    python mtt_preflight.py                 # env + capability + binding. No kernel.
    python mtt_preflight.py --run prefill   # one minimal prefill launch
    python mtt_preflight.py --run decode    # one minimal decode launch

Run the bare form FIRST and send the output back before running any kernel.
"""

import argparse
import hashlib
import os
import sys
import traceback

W = 78


def hdr(t):
    print("\n" + "=" * W + f"\n  {t}\n" + "=" * W)


def row(k, v):
    print(f"  {k:<34} {v}")


def check(label, fn):
    """Run one probe; never let it abort the rest of the report."""
    try:
        row(label, fn())
    except Exception as e:  # noqa: BLE001 - a probe failing IS the datapoint
        row(label, f"!! {type(e).__name__}: {e}")


def md5(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError as e:
        return f"<unreadable: {e.strerror}>"


# --------------------------------------------------------------------------
# 1. interpreter + frameworks
# --------------------------------------------------------------------------
def stage_env():
    hdr("1. INTERPRETER / FRAMEWORKS")
    row("python", sys.version.split()[0])
    row("executable", sys.executable)

    def _torch():
        import torch

        return torch.__version__

    check("torch", _torch)

    def _musa():
        import torch
        import torch_musa  # noqa: F401

        v = getattr(torch_musa, "__version__", "?")
        return f"{v}   musa.is_available={torch.musa.is_available()}"

    check("torch_musa", _musa)

    def _triton():
        import triton

        return f"{triton.__version__}   @ {os.path.dirname(triton.__file__)}"

    check("triton / flagtree", _triton)


# --------------------------------------------------------------------------
# 2. THE known killer: which llc actually runs
# --------------------------------------------------------------------------
def stage_llc():
    hdr("2. llc RESOLUTION  <-- the known S5000 killer")
    print(
        "  FlagTree ships a fixed llc at backends/mthreads/bin/llc (md5 cec9ff66...).\n"
        "  The MUSA 4.3.5 system llc (md5 7d0a51d8...) aborts in MTGPU ISel\n"
        "  (SelectionDAGISel::CannotYetSelect). BOTH self-report LLVM 14.0.0, so a\n"
        "  version string cannot tell them apart -- only the md5 can.\n"
        "  tools/setup.sh defaults to FLAGTREE_VERSION=0.6.0, which has NO bin/ and\n"
        "  therefore silently falls back to the broken system llc.\n"
    )

    bundled = None
    try:
        import triton

        bundled = os.path.join(
            os.path.dirname(triton.__file__), "backends", "mthreads", "bin", "llc"
        )
    except Exception as e:  # noqa: BLE001
        row("bundled llc", f"!! could not locate triton: {e}")

    if bundled:
        if os.path.isfile(bundled):
            m = md5(bundled)
            good = m.startswith("cec9ff66")
            row("bundled llc", bundled)
            row("  md5", f"{m}   {'<-- GOOD' if good else '<-- UNEXPECTED'}")
        else:
            row("bundled llc", f"MISSING at {bundled}")
            row("  verdict", "!! will fall back to system llc -- expect abort")

    sys_llc = os.environ.get("MUSA_HOME", "/usr/local/musa") + "/bin/llc"
    if os.path.isfile(sys_llc):
        m = md5(sys_llc)
        row("system llc", sys_llc)
        row("  md5", f"{m}   {'<-- KNOWN BROKEN' if m.startswith('7d0a51d8') else ''}")
    else:
        row("system llc", f"not present at {sys_llc}")


# --------------------------------------------------------------------------
# 3. TLE: importable is NOT the same as lowerable
# --------------------------------------------------------------------------
def stage_tle():
    hdr("3. TLE AVAILABILITY  <-- decides which code path the op takes")
    print(
        "  has_triton_tle() only checks that the IMPORT succeeds. It does not check\n"
        "  that the MUSA backend can lower tle.gpu.alloc / local_ptr to shared\n"
        "  memory. If the import works but lowering does not, the op takes the TLE\n"
        "  path and fails -- while the non-TLE fallback would have worked.\n"
    )

    def _hastle():
        from flaggems_vllm.utils.triton_version_utils import has_triton_tle

        return f"has_triton_tle(3,6,0) = {has_triton_tle(3, 6, 0)}"

    check("version+import gate", _hastle)

    def _syms():
        import triton.experimental.tle.language as tle

        have = []
        for path in ("gpu.alloc", "gpu.local_ptr", "gpu.smem", "cumsum"):
            obj = tle
            ok = True
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            have.append(f"{path}={'yes' if ok else 'NO'}")
        return "  ".join(have)

    check("tle symbols present", _syms)

    # What the op itself will actually do -- read it off the op's own globals.
    for op in ("top_k_per_row_prefill", "top_k_per_row_decode"):

        def _path(op=op):
            import flaggems_vllm

            g = getattr(flaggems_vllm, op).__globals__
            tle_on = g.get("HAS_TLE")
            branch = "TLE (smem)" if tle_on else "non-TLE (global scratch)"
            return f"HAS_TLE={tle_on}  ->  {branch}"

        check(f"{op} branch", _path)


# --------------------------------------------------------------------------
# 4. device geometry vs the op's hardcoded launch
# --------------------------------------------------------------------------
def stage_device():
    hdr("4. DEVICE GEOMETRY vs THE OP'S FIXED LAUNCH")
    print(
        "  The op hardcodes NUM_THREADS_PER_BLOCK=512 and num_warps=512//32=16.\n"
        "  On fused_v4 this card strongly preferred num_warps=1; num_warps=4 was a\n"
        "  2.6x REGRESSION. 16 warps is far outside anything measured here.\n"
        "  Read elements-per-lane and program width, never a (TPP,warps) pair\n"
        "  carried over from another card.\n"
    )

    def _props():
        import torch
        import torch_musa  # noqa: F401

        p = torch.musa.get_device_properties(0)
        bits = [f"name={p.name}"]
        for a in (
            "warp_size",
            "max_threads_per_block",
            "shared_memory_per_block",
            "multi_processor_count",
        ):
            if hasattr(p, a):
                bits.append(f"{a}={getattr(p, a)}")
        return "\n" + "\n".join(f"      {b}" for b in bits)

    check("musa device properties", _props)
    row("op smem need (TLE path)", "prefill ~21 KB / decode ~25 KB @ top_k=1024")
    row("", "top_k=2048 -> ~29 / ~33 KB. Ceiling 128 KB: ample headroom.")


# --------------------------------------------------------------------------
# 5. which implementation is bound
# --------------------------------------------------------------------------
def stage_binding():
    hdr("5. BINDING: generic or vendor override?")

    def _vendor():
        import flaggems_vllm

        return f"{flaggems_vllm.vendor_name}   device={flaggems_vllm.device}"

    check("vendor / device", _vendor)

    for op in ("top_k_per_row_prefill", "top_k_per_row_decode"):

        def _bound(op=op):
            import importlib

            import flaggems_vllm

            top = getattr(flaggems_vllm, op)
            generic = getattr(importlib.import_module(f"flaggems_vllm.ops.{op}"), op)
            which = "OVERRIDE" if top is not generic else "generic"
            return f"{which:<9} from {top.__module__}"

        check(f"{op}", _bound)

    print(
        "\n  Expected on MTT today: 'generic' for both -- no mthreads override exists\n"
        "  in either repo. MetaX and Hygon are the only prefill overrides ported.\n"
    )


# --------------------------------------------------------------------------
# 6. opt-in minimal kernel launch
# --------------------------------------------------------------------------
def run_kernel(which):
    hdr(f"6. MINIMAL LAUNCH: {which}")
    import torch

    import flaggems_vllm

    dev = flaggems_vllm.device
    torch.manual_seed(0)

    num_rows, vocab, top_k = 1, 20000, 1024
    row("shape", f"num_rows={num_rows} vocab={vocab} top_k={top_k} (smallest legal)")

    logits = torch.randn((num_rows, vocab), dtype=torch.float32, device=dev)
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=dev)

    if which == "prefill":
        starts = torch.zeros((num_rows,), dtype=torch.int32, device=dev)
        ends = torch.full((num_rows,), vocab, dtype=torch.int32, device=dev)
        flaggems_vllm.top_k_per_row_prefill(
            logits, starts, ends, indices, num_rows,
            logits.stride(0), logits.stride(1), top_k,
        )
    else:
        # next_n is a plain Python int -- "number of next tokens (unused, kept for
        # API compatibility)". It is NOT a tensor: passing one makes it a
        # pointer<int32>, and the kernel dies at compile time on `row_id // next_n`
        # with IncompatibleTypeErrorImpl. Only seq_lens is a device tensor.
        next_n = 1
        seq_lens = torch.tensor([vocab], dtype=torch.int32, device=dev)
        flaggems_vllm.top_k_per_row_decode(
            logits, next_n, seq_lens, indices, num_rows,
            logits.stride(0), logits.stride(1), top_k,
        )

    torch.musa.synchronize()

    # Correctness against torch, on the SEEDED input above. (An unseeded harness
    # once made 17 provably-equivalent kernels all look wrong -- don't repeat it.)
    ref = torch.topk(logits[0, :vocab], top_k, largest=True, sorted=False).indices
    got = indices[0].to(torch.int64)
    same = torch.equal(
        torch.sort(logits[0][got]).values, torch.sort(logits[0][ref]).values
    )
    row("launch", "OK")
    row("selected values match torch", "YES" if same else "NO  <-- accuracy bug")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["prefill", "decode"], default=None)
    args = ap.parse_args()

    print("=" * W)
    print("  MTT S5000 preflight -- top_k_per_row_{prefill,decode}")
    print("=" * W)

    stage_env()
    stage_llc()
    stage_tle()
    stage_device()
    stage_binding()

    if args.run:
        try:
            run_kernel(args.run)
        except Exception:  # noqa: BLE001
            hdr(f"6. MINIMAL LAUNCH: {args.run}  -- FAILED")
            traceback.print_exc()
            print(
                "\n  MUSA errors are STICKY: this context is now poisoned. Re-run in a\n"
                "  FRESH process for the other op, and do not trust any later result\n"
                "  from this one.\n"
            )
            sys.exit(1)
    else:
        hdr("NO KERNEL LAUNCHED")
        print(
            "  Send this output back before running anything on the card.\n"
            "  Then, in separate processes:\n"
            "      python mtt_preflight.py --run prefill\n"
            "      python mtt_preflight.py --run decode\n"
        )


if __name__ == "__main__":
    main()
