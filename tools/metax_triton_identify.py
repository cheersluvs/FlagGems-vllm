"""Which Triton is installed here, and does it carry the metax TLE plugin?

The operator's own call to tle.gpu.alloc (nv_mma_shared_layout=False) fails on
a missing `make_swizzled_shared_encoding_attr` -- and that is precisely the
binding FlagTree's third_party/metax/plugin/mctle registers, alongside
create_local_alloc / create_local_load / create_local_pointers /
create_local_store. So FlagTree upstream supports metax TLE and this build does
not have it, which makes it a packaging question rather than a capability one.

Pin down what is actually installed, so the ask can name a version rather than
a wish.

    python tools/metax_triton_identify.py
"""

import os
import pathlib
import sys

import triton


def main():
    root = pathlib.Path(triton.__file__).parent
    print(f"triton {triton.__version__}")
    print(f"  at {root}")
    for k in ("__flagtree_version__", "__flagtree__", "__vendor__"):
        if hasattr(triton, k):
            print(f"  {k} = {getattr(triton, k)}")

    print("\n--- distribution metadata")
    try:
        from importlib.metadata import distributions
        for d in distributions():
            n = (d.metadata["Name"] or "").lower()
            if "triton" in n or "flagtree" in n:
                print(f"  {d.metadata['Name']} {d.version}")
    except Exception as exc:  # noqa: BLE001
        print(f"  unavailable: {exc}")

    print("\n--- backends present")
    bdir = root / "backends"
    if bdir.is_dir():
        for p in sorted(bdir.iterdir()):
            if p.is_dir():
                extras = [f.name for f in p.iterdir()
                          if f.name in ("tle_supported.py", "compiler.py", "driver.py")]
                print(f"  {p.name:<14} {' '.join(sorted(extras))}")
    else:
        print("  no backends/ directory")

    print("\n--- does the TLE python package ship?")
    tle = root / "experimental" / "tle"
    print(f"  {tle}: {'yes' if tle.is_dir() else 'NO'}")

    print("\n--- what libtriton's builder exposes that looks TLE-ish")
    try:
        from triton._C import libtriton
        b = [n for n in dir(libtriton.ir.builder) if not n.startswith("_")]
        want = ("make_swizzled_shared_encoding_attr",
                "make_nv_mma_shared_encoding_attr",
                "create_local_pointers", "create_local_alloc",
                "create_local_load", "create_local_store",
                "create_exclusive_cumsum")
        for w in want:
            print(f"  {w:<38} {'YES' if w in b else 'no'}")
        near = sorted(n for n in b
                      if any(t in n for t in ("local", "shared", "cumsum", "tle")))
        print(f"  anything else nearby: {near if near else 'none'}")
        print(f"  libtriton has a `tle` submodule: "
              f"{'yes' if hasattr(libtriton, 'tle') else 'no'}")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not inspect: {type(exc).__name__}: {exc}")

    print("\n--- environment that might select a backend")
    for k in sorted(os.environ):
        if "TRITON" in k or "FLAGTREE" in k or "MACA" in k:
            print(f"  {k}={os.environ[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
